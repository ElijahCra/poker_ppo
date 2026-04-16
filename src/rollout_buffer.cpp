#include "rollout_buffer.h"

namespace poker_ppo {

RolloutBuffer::RolloutBuffer(int num_steps, int num_envs,
                             int obs_dim, int action_count)
    : num_steps_(num_steps), num_envs_(num_envs),
      obs_dim_(obs_dim), action_count_(action_count)
{
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt64);

    obs_         = torch::zeros({num_steps, num_envs, obs_dim}, opts_f);
    actions_     = torch::zeros({num_steps, num_envs}, opts_i);
    log_probs_   = torch::zeros({num_steps, num_envs}, opts_f);
    rewards_     = torch::zeros({num_steps, num_envs}, opts_f);
    dones_       = torch::zeros({num_steps, num_envs}, opts_f);
    values_      = torch::zeros({num_steps, num_envs}, opts_f);
    legal_masks_ = torch::zeros({num_steps, num_envs, action_count}, opts_f);
    advantages_  = torch::zeros({num_steps, num_envs}, opts_f);
    returns_     = torch::zeros({num_steps, num_envs}, opts_f);
}

void RolloutBuffer::insert(int step,
                           torch::Tensor obs,
                           torch::Tensor actions,
                           torch::Tensor log_probs,
                           torch::Tensor rewards,
                           torch::Tensor dones,
                           torch::Tensor values,
                           torch::Tensor legal_masks) {
    obs_[step]         = obs;
    actions_[step]     = actions;
    log_probs_[step]   = log_probs;
    rewards_[step]     = rewards;
    dones_[step]       = dones;
    values_[step]      = values;
    legal_masks_[step] = legal_masks;
}

void RolloutBuffer::compute_returns(torch::Tensor next_value,
                                    torch::Tensor next_done,
                                    float gamma, float gae_lambda) {
    auto lastgaelam = torch::zeros({num_envs_});
    for (int t = num_steps_ - 1; t >= 0; --t) {
        torch::Tensor next_nonterminal, next_values;
        if (t == num_steps_ - 1) {
            next_nonterminal = 1.0f - next_done;
            next_values      = next_value;
        } else {
            next_nonterminal = 1.0f - dones_[t + 1];
            next_values      = values_[t + 1];
        }
        auto delta  = rewards_[t] + gamma * next_values * next_nonterminal
                      - values_[t];
        lastgaelam  = delta + gamma * gae_lambda * next_nonterminal * lastgaelam;
        advantages_[t] = lastgaelam;
    }
    returns_ = advantages_ + values_;
}

RolloutBuffer::FlatBatch RolloutBuffer::flatten() const {
    int B = num_steps_ * num_envs_;
    return {
        obs_.reshape({B, obs_dim_}),
        actions_.reshape({B}),
        log_probs_.reshape({B}),
        advantages_.reshape({B}),
        returns_.reshape({B}),
        values_.reshape({B}),
        legal_masks_.reshape({B, action_count_}),
    };
}

} // namespace poker_ppo
