#pragma once

#include "types.h"
#include <torch/torch.h>
#include <vector>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// RolloutBuffer  — stores one rollout of (num_steps × num_envs) transitions
// ─────────────────────────────────────────────────────────────────────────────
//
// In addition to the standard PPO fields, this buffer tracks which player
// (seat 0 or 1) is acting at each step.  This is needed for correct GAE
// computation in two-player zero-sum self-play: when the acting player
// changes between consecutive steps, the bootstrapped value must be negated
// because the opponent's expected future reward is the negative of ours.

class RolloutBuffer {
public:
    RolloutBuffer(int num_steps, int num_envs, int obs_dim, int action_count);

    /// Insert a single time-step across all envs.
    void insert(int step,
                torch::Tensor obs,              // [num_envs, obs_dim]
                torch::Tensor actions,           // [num_envs]
                torch::Tensor log_probs,         // [num_envs]
                torch::Tensor rewards,           // [num_envs]
                torch::Tensor dones,             // [num_envs]
                torch::Tensor values,            // [num_envs]
                torch::Tensor legal_masks,       // [num_envs, action_count]
                torch::Tensor current_players);  // [num_envs]  int32 (0 or 1)

    /// Compute GAE returns after a full rollout.
    /// next_value:  [num_envs]  — bootstrap value from critic at t=num_steps.
    /// next_done:   [num_envs]  — done flag at t=num_steps.
    /// next_player: [num_envs]  — acting player at t=num_steps.
    ///
    /// When the acting player changes between steps t and t+1, the
    /// bootstrapped value and advantage carry-over are negated (zero-sum).
    void compute_returns(torch::Tensor next_value,
                         torch::Tensor next_done,
                         torch::Tensor next_player,
                         float gamma, float gae_lambda);

    /// Flatten and return a batch for training.
    struct FlatBatch {
        torch::Tensor obs;          // [B, obs_dim]
        torch::Tensor actions;      // [B]
        torch::Tensor log_probs;    // [B]
        torch::Tensor advantages;   // [B]
        torch::Tensor returns;      // [B]
        torch::Tensor values;       // [B]
        torch::Tensor legal_masks;  // [B, action_count]
    };
    FlatBatch flatten() const;

    int num_steps()  const { return num_steps_; }
    int num_envs()   const { return num_envs_; }

private:
    int num_steps_, num_envs_, obs_dim_, action_count_;

    // Storage: [num_steps, num_envs, ...]
    torch::Tensor obs_;              // [T, N, obs_dim]
    torch::Tensor actions_;          // [T, N]
    torch::Tensor log_probs_;        // [T, N]
    torch::Tensor rewards_;          // [T, N]
    torch::Tensor dones_;            // [T, N]
    torch::Tensor values_;           // [T, N]
    torch::Tensor legal_masks_;      // [T, N, action_count]
    torch::Tensor current_players_;  // [T, N]  int32

    // Computed after rollout
    torch::Tensor advantages_;       // [T, N]
    torch::Tensor returns_;          // [T, N]
};

} // namespace poker_ppo
