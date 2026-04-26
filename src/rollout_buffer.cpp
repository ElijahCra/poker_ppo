#include "rollout_buffer.h"

#include <algorithm>

namespace poker_ppo {

RolloutBuffer::RolloutBuffer(int num_steps, int num_envs,
                             int obs_dim, int action_count,
                             torch::Device device)
    : num_steps_(num_steps), num_envs_(num_envs),
      obs_dim_(obs_dim), action_count_(action_count),
      device_(device)
{
    // Storage lives on CPU: pushes are driven from per-env worker threads and
    // doing device writes per transition would serialise on CUDA's launch
    // queue. One batched CPU→device copy happens in flatten() instead.
    auto cpu_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto cpu_i = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

    for (int p = 0; p < 2; ++p) {
        obs_[p]         = torch::zeros({num_steps, num_envs, obs_dim},     cpu_f);
        actions_[p]     = torch::zeros({num_steps, num_envs},              cpu_i);
        log_probs_[p]   = torch::zeros({num_steps, num_envs},              cpu_f);
        rewards_[p]     = torch::zeros({num_steps, num_envs},              cpu_f);
        dones_[p]       = torch::zeros({num_steps, num_envs},              cpu_f);
        values_[p]      = torch::zeros({num_steps, num_envs},              cpu_f);
        legal_masks_[p] = torch::zeros({num_steps, num_envs, action_count}, cpu_f);
        advantages_[p]  = torch::zeros({num_steps, num_envs},              cpu_f);
        returns_[p]     = torch::zeros({num_steps, num_envs},              cpu_f);
        counts_[p].assign(num_envs, 0);
    }
}

void RolloutBuffer::clear() {
    for (int p = 0; p < 2; ++p) {
        std::fill(counts_[p].begin(), counts_[p].end(), 0);
    }
}

void RolloutBuffer::push(int player, int env_idx,
                         torch::Tensor obs,
                         int64_t action,
                         float log_prob,
                         float reward,
                         float done,
                         float value,
                         torch::Tensor mask) {
    const int t = counts_[player][env_idx]++;

    // obs / mask may arrive on a non-CPU device (e.g. when the serial
    // collector keeps state on CUDA); force CPU for buffer storage.
    if (obs.device()  != torch::kCPU) obs  = obs.to(torch::kCPU);
    if (mask.device() != torch::kCPU) mask = mask.to(torch::kCPU);

    obs_[player][t][env_idx]         = obs;
    legal_masks_[player][t][env_idx] = mask;
    actions_[player]  .accessor<int64_t, 2>()[t][env_idx] = action;
    log_probs_[player].accessor<float,   2>()[t][env_idx] = log_prob;
    rewards_[player]  .accessor<float,   2>()[t][env_idx] = reward;
    dones_[player]    .accessor<float,   2>()[t][env_idx] = done;
    values_[player]   .accessor<float,   2>()[t][env_idx] = value;
}

void RolloutBuffer::compute_returns(
    float gamma, float lam,
    const torch::Tensor& bootstrap_values,
    const torch::Tensor& bootstrap_terminal) {

    TORCH_CHECK(bootstrap_values.dim() == 2 &&
                bootstrap_values.size(0) == 2 &&
                bootstrap_values.size(1) == num_envs_,
                "bootstrap_values must be [2, num_envs]");
    TORCH_CHECK(bootstrap_terminal.sizes() == bootstrap_values.sizes(),
                "bootstrap_terminal shape must match bootstrap_values");

    auto bv = bootstrap_values.to(torch::kCPU).contiguous();
    auto bt = bootstrap_terminal.to(torch::kCPU).contiguous();
    auto bv_a = bv.accessor<float, 2>();
    auto bt_a = bt.accessor<float, 2>();

    for (int p = 0; p < 2; ++p) {
        advantages_[p].zero_();
        returns_[p].zero_();

        auto rewards_a = rewards_[p].accessor<float, 2>();
        auto dones_a   = dones_[p].accessor<float, 2>();
        auto values_a  = values_[p].accessor<float, 2>();
        auto advs_a    = advantages_[p].accessor<float, 2>();

        for (int e = 0; e < num_envs_; ++e) {
            const int T = counts_[p][e];
            if (T == 0) continue;
            float lastgae = 0.0f;
            for (int t = T - 1; t >= 0; --t) {
                float next_nonterminal, next_value;
                if (t == T - 1) {
                    if (bt_a[p][e] > 0.5f) {
                        // Tail was followed by an episode terminal — V_next = 0.
                        next_nonterminal = 0.0f;
                        next_value       = 0.0f;
                    } else {
                        // Tail was truncated by rollout-end — bootstrap from
                        // the trainer-supplied V at the carry obs.
                        next_nonterminal = 1.0f;
                        next_value       = bv_a[p][e];
                    }
                } else {
                    next_nonterminal = 1.0f - dones_a[t + 1][e];
                    next_value       = values_a[t + 1][e];
                }
                const float delta =
                    rewards_a[t][e] + gamma * next_value * next_nonterminal
                    - values_a[t][e];
                lastgae = delta + gamma * lam * next_nonterminal * lastgae;
                advs_a[t][e] = lastgae;
            }
        }
        returns_[p] = advantages_[p] + values_[p];
    }
}

RolloutBuffer::FlatBatch RolloutBuffer::flatten() const {
    std::vector<torch::Tensor> obs_list, actions_list, logp_list;
    std::vector<torch::Tensor> advs_list, rets_list, vals_list, masks_list;

    for (int p = 0; p < 2; ++p) {
        for (int e = 0; e < num_envs_; ++e) {
            const int T = counts_[p][e];
            if (T == 0) continue;
            obs_list    .push_back(obs_[p]        .slice(0, 0, T).select(1, e));
            actions_list.push_back(actions_[p]    .slice(0, 0, T).select(1, e));
            logp_list   .push_back(log_probs_[p]  .slice(0, 0, T).select(1, e));
            advs_list   .push_back(advantages_[p] .slice(0, 0, T).select(1, e));
            rets_list   .push_back(returns_[p]    .slice(0, 0, T).select(1, e));
            vals_list   .push_back(values_[p]     .slice(0, 0, T).select(1, e));
            masks_list  .push_back(legal_masks_[p].slice(0, 0, T).select(1, e));
        }
    }

    auto to_dev = [this](torch::Tensor t) {
        return t.to(device_, /*non_blocking=*/false, /*copy=*/false);
    };

    if (obs_list.empty()) {
        auto f = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        auto i = torch::TensorOptions().dtype(torch::kInt64).device(device_);
        return {
            torch::zeros({0, obs_dim_},      f),
            torch::zeros({0},                i),
            torch::zeros({0},                f),
            torch::zeros({0},                f),
            torch::zeros({0},                f),
            torch::zeros({0},                f),
            torch::zeros({0, action_count_}, f),
        };
    }
    return {
        to_dev(torch::cat(obs_list,     0)),
        to_dev(torch::cat(actions_list, 0)),
        to_dev(torch::cat(logp_list,    0)),
        to_dev(torch::cat(advs_list,    0)),
        to_dev(torch::cat(rets_list,    0)),
        to_dev(torch::cat(vals_list,    0)),
        to_dev(torch::cat(masks_list,   0)),
    };
}

} // namespace poker_ppo
