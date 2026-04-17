#include "rollout_buffer.h"

namespace poker_ppo {

RolloutBuffer::RolloutBuffer(int num_steps, int num_envs,
                             int obs_dim, int action_count)
    : num_steps_(num_steps), num_envs_(num_envs),
      obs_dim_(obs_dim), action_count_(action_count)
{
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt64);

    obs_             = torch::zeros({num_steps, num_envs, obs_dim}, opts_f);
    actions_         = torch::zeros({num_steps, num_envs}, opts_i);
    log_probs_       = torch::zeros({num_steps, num_envs}, opts_f);
    rewards_         = torch::zeros({num_steps, num_envs}, opts_f);
    dones_           = torch::zeros({num_steps, num_envs}, opts_f);
    values_          = torch::zeros({num_steps, num_envs}, opts_f);
    legal_masks_     = torch::zeros({num_steps, num_envs, action_count}, opts_f);
    current_players_ = torch::zeros({num_steps, num_envs},
                                    torch::TensorOptions().dtype(torch::kInt32));
    advantages_      = torch::zeros({num_steps, num_envs}, opts_f);
    returns_         = torch::zeros({num_steps, num_envs}, opts_f);
}

void RolloutBuffer::insert(int step,
                           torch::Tensor obs,
                           torch::Tensor actions,
                           torch::Tensor log_probs,
                           torch::Tensor rewards,
                           torch::Tensor dones,
                           torch::Tensor values,
                           torch::Tensor legal_masks,
                           torch::Tensor current_players) {
    obs_[step]             = obs;
    actions_[step]         = actions;
    log_probs_[step]       = log_probs;
    rewards_[step]         = rewards;
    dones_[step]           = dones;
    values_[step]          = values;
    legal_masks_[step]     = legal_masks;
    current_players_[step] = current_players;
}

void RolloutBuffer::compute_returns(torch::Tensor next_value,
                                    torch::Tensor next_done,
                                    torch::Tensor next_player,
                                    float gamma, float gae_lambda) {
    // ─────────────────────────────────────────────────────────────────────
    // Zero-sum GAE for alternating self-play
    // ─────────────────────────────────────────────────────────────────────
    //
    // Standard GAE:
    //   delta_t = r_t + γ * V(s_{t+1}) * (1-d_{t+1}) - V(s_t)
    //   A_t     = delta_t + γ * λ * (1-d_{t+1}) * A_{t+1}
    //
    // Problem: V(s_{t+1}) is estimated from the perspective of whoever
    // acts at t+1.  In zero-sum self-play, if the player switches, this
    // value is the *opponent's* expected future reward, which equals the
    // negative of our expected future reward.
    //
    // Fix: multiply next_value and A_{t+1} by a sign factor:
    //   sign = +1 if same player acts at t and t+1
    //   sign = -1 if the player switches
    //
    // When d_{t+1} = 1 (episode boundary), the terms are zeroed out
    // anyway, so the sign doesn't matter at episode boundaries.
    // ─────────────────────────────────────────────────────────────────────

    auto lastgaelam = torch::zeros({num_envs_});

    for (int t = num_steps_ - 1; t >= 0; --t) {
        torch::Tensor next_nonterminal, next_values, next_p;

        if (t == num_steps_ - 1) {
            next_nonterminal = 1.0f - next_done;
            next_values      = next_value;
            next_p           = next_player;
        } else {
            next_nonterminal = 1.0f - dones_[t + 1];
            next_values      = values_[t + 1];
            next_p           = current_players_[t + 1];
        }

        // sign = +1 if same player, -1 if player switches
        auto same_player = (current_players_[t] == next_p).to(torch::kFloat32);
        auto sign = 2.0f * same_player - 1.0f;   // maps {0,1} → {-1,+1}

        auto delta = rewards_[t]
                   + gamma * sign * next_values * next_nonterminal
                   - values_[t];

        lastgaelam = delta
                   + gamma * gae_lambda * sign * next_nonterminal * lastgaelam;

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
