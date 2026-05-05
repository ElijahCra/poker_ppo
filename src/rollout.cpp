#include "rollout.h"

#include "opponent_manager.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace poker_ppo {

RolloutBuffer::RolloutBuffer(int num_steps, int num_envs,
                             int obs_dim, int action_count,
                             torch::Device device)
    : num_steps_(num_steps), num_envs_(num_envs),
      obs_dim_(obs_dim), action_count_(action_count),
      device_(device)
{
    // Storage on CPU: pushes come from per-env worker threads and per-
    // transition device writes would serialise on CUDA's launch queue.
    // One batched H2D copy in flatten().
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

    // Serial collector keeps state on device; force CPU for storage.
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
                        // Terminal: V_next = 0.
                        next_nonterminal = 0.0f;
                        next_value       = 0.0f;
                    } else {
                        // Truncation: bootstrap from carry_obs.
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

namespace {

struct BootstrapTensors {
    torch::Tensor values;     // [2, N] float32 CPU
    torch::Tensor terminal;   // [2, N] float32 CPU
};

// Critic-forwards cur_obs (next acting player's view), then zero-sum
// negates for the other player to fill compute_returns' [2, N] tensors.
BootstrapTensors build_bootstrap(
    ActorCritic&                            network,
    const torch::Tensor&                    cur_obs,           // [N, D] device
    const torch::Tensor&                    cur_player_cpu,    // [N] int32 CPU
    const std::vector<PlayerRolloutState>&  player_state)
{
    const int N = static_cast<int>(cur_obs.size(0));
    auto f_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // NoGrad is in scope at the call site.
    auto V_carry = network->get_value(cur_obs).detach().to(torch::kCPU).contiguous();

    auto values   = torch::zeros({2, N}, f_cpu);
    auto terminal = torch::zeros({2, N}, f_cpu);

    auto v_acc  = V_carry.accessor<float, 1>();
    auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();
    auto bv_acc = values.accessor<float, 2>();
    auto bt_acc = terminal.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        const int   next_actor = cp_acc[i];
        const float v          = v_acc[i];
        bv_acc[next_actor][i]     = v;
        bv_acc[1 - next_actor][i] = -v;
        bt_acc[0][i] = player_state[i].tail_was_terminal[0] ? 1.0f : 0.0f;
        bt_acc[1][i] = player_state[i].tail_was_terminal[1] ? 1.0f : 0.0f;
    }
    return {values, terminal};
}

}  // namespace

RolloutCollector::RolloutCollector(IPokerEnvironmentFactory& factory,
                                   const BetConfig&          bet_cfg,
                                   int                       num_envs,
                                   int                       num_steps,
                                   torch::Device             device)
    : device_(device), num_envs_(num_envs), num_steps_(num_steps)
{
    vec_env_ = std::make_unique<VectorizedEnv>(factory, bet_cfg, num_envs);
    const int obs_dim      = vec_env_->obs_dim();
    const int action_count = vec_env_->action_count();
    buffer_ = std::make_unique<RolloutBuffer>(
        num_steps, num_envs, obs_dim, action_count, device_);
}

RolloutCollector::~RolloutCollector() = default;

void RolloutCollector::ensure_step_pool() {
    if (step_pool_) return;
    const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int n  = std::min(num_envs_, hw);
    step_pool_   = std::make_unique<StepThreadPool>(n);
}

void RolloutCollector::init_carry() {
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();
    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    carry_obs_            = torch::zeros({num_envs_, D}, f_cpu);
    carry_legal_mask_     = torch::zeros({num_envs_, A}, f_cpu);
    carry_current_player_ = torch::zeros({num_envs_},    i32_cpu);
    carry_done_           = torch::zeros({num_envs_},    f_cpu);

    vec_env_->reset_all();
    for (int i = 0; i < num_envs_; ++i) {
        carry_obs_[i]        = vec_env_->env(i).observation();
        carry_legal_mask_[i] = vec_env_->env(i).legal_action_mask();
        carry_current_player_.accessor<int32_t, 1>()[i] =
            static_cast<int32_t>(vec_env_->env(i).current_player());
    }
}

void RolloutCollector::collect(Strategy         strategy,
                               ActorCritic&     network,
                               OpponentManager& opp_mgr,
                               int              update_idx,
                               float            gamma,
                               float            gae_lambda)
{
    network->eval();
    // InferenceMode is stricter than NoGradGuard (no view tracking, no
    // version counter) — slightly faster and safe here because rollout
    // tensors are detached/copied to CPU before storage in the buffer.
    c10::InferenceMode inference_guard;

    if (strategy == Strategy::Threadpool) ensure_step_pool();

    buffer_->clear();
    opp_mgr.prepare_rollout(update_idx);

    auto& envs = vec_env_->envs_mut();
    const int N = num_envs_;
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();

    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    // Seed from carry_done_ so the first transition after a previous-
    // rollout reset carries the new-episode marker.
    std::vector<PlayerRolloutState> player_state(N);
    for (int i = 0; i < N; ++i) {
        const float d = carry_done_.accessor<float, 1>()[i];
        player_state[i].next_done_flag[0] = d;
        player_state[i].next_done_flag[1] = d;
    }

    auto cur_obs    = carry_obs_.to(device_);
    auto cur_mask   = carry_legal_mask_.to(device_);
    auto cur_player = carry_current_player_.to(device_);

    std::vector<uint8_t> did_reset(N, 0);

    for (int step = 0; step < num_steps_; ++step) {
        // One full-batch forward per step regardless of strategy.
        auto ar = network->get_action(cur_obs, cur_mask);

        auto actions_cpu    = ar.action.to(torch::kCPU).contiguous();
        auto logp_cpu       = ar.log_prob.to(torch::kCPU).contiguous();
        auto value_cpu      = ar.value.to(torch::kCPU).contiguous();
        auto cur_obs_cpu    = cur_obs.to(torch::kCPU).contiguous();
        auto cur_mask_cpu   = cur_mask.to(torch::kCPU).contiguous();
        auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();

        opp_mgr.apply_action_overrides(
            cur_obs, cur_mask, cur_player_cpu, actions_cpu);

        auto a_acc  = actions_cpu.accessor<int64_t, 1>();
        auto lp_acc = logp_cpu.accessor<float, 1>();
        auto v_acc  = value_cpu.accessor<float, 1>();
        auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();

        auto next_obs    = torch::zeros({N, D}, f_cpu);
        auto next_mask   = torch::zeros({N, A}, f_cpu);
        auto next_player = torch::zeros({N},    i32_cpu);
        auto np_acc      = next_player.accessor<int32_t, 1>();

        std::fill(did_reset.begin(), did_reset.end(), 0);

        // Per-i state is disjoint, so step_body is safe to parallelise
        // across i. Buffer pushes at (player, i) are also disjoint.
        auto step_body = [&](int i) {
            const int acting = cp_acc[i];

            if (opp_mgr.should_record(i, acting)) {
                player_state[i].record_step(
                    acting, i, *buffer_,
                    cur_obs_cpu[i], cur_mask_cpu[i],
                    a_acc[i], lp_acc[i], v_acc[i]);
            }

            auto res = envs[i]->step(static_cast<int>(a_acc[i]));
            player_state[i].step_reward(res.reward);

            if (res.done) {
                player_state[i].flush_on_terminal(i, *buffer_);
                auto rr = envs[i]->reset();
                next_obs[i]  = rr.observation;
                next_mask[i] = rr.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
                did_reset[i] = 1;
            } else {
                next_obs[i]  = res.observation;
                next_mask[i] = res.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
            }
        };

        if (strategy == Strategy::Threadpool) {
            step_pool_->parallel_for(N, step_body);
        } else {
            for (int i = 0; i < N; ++i) step_body(i);
        }

        // Serial: opp_mgr's RNG isn't thread-safe.
        for (int i = 0; i < N; ++i) {
            if (did_reset[i]) opp_mgr.on_episode_terminal(i, update_idx);
        }

        cur_obs    = next_obs.to(device_);
        cur_mask   = next_mask.to(device_);
        cur_player = next_player.to(device_);
    }

    // Truncations, not terminals — bootstrapped from V(carry_obs).
    for (int i = 0; i < N; ++i) {
        player_state[i].flush_on_rollout_end(i, *buffer_);
    }

    auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();
    auto bootstrap = build_bootstrap(network, cur_obs, cur_player_cpu, player_state);
    buffer_->compute_returns(gamma, gae_lambda,
                             bootstrap.values, bootstrap.terminal);

    carry_obs_            = cur_obs.to(torch::kCPU);
    carry_legal_mask_     = cur_mask.to(torch::kCPU);
    carry_current_player_ = cur_player_cpu;
    // Pending transitions are already flushed; clear the boundary state.
    carry_done_.zero_();

    global_step_ += num_envs_ * num_steps_;
}

} // namespace poker_ppo
