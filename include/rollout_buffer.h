#pragma once

#include "types.h"
#include <torch/torch.h>
#include <vector>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// RolloutBuffer  — per-player rollout storage for alternating self-play
// ─────────────────────────────────────────────────────────────────────────────
//
// Faithful to OpenSpiel PPO (Timbers et al., App. G.5): every env transition
// is pushed into the buffer for the player who acted at that step, and each
// (player, env) trajectory has vanilla GAE computed on it independently.
//
// Reward attribution: when the env emits a zero-sum reward r (in player-0's
// frame), the caller accumulates +r for player 0 and -r for player 1 between
// the player's consecutive actions. At the player's next action the
// accumulated reward becomes the reward of the *previous* transition.
//
// Rollout-end truncation: the last recorded transition per (player, env)
// trajectory is treated as terminal (no bootstrap). The bias is ≤ 1 tail
// transition per player per env per rollout and decays out in subsequent
// updates.

class RolloutBuffer {
public:
    RolloutBuffer(int num_steps, int num_envs, int obs_dim, int action_count,
                  torch::Device device = torch::kCPU);

    /// Reset per-(player, env) transition counts. Storage tensors are reused.
    void clear();

    /// Append one transition for (player, env_idx). Thread-safe as long as at
    /// most one thread writes to each (player, env_idx) slot at a time —
    /// naturally satisfied by one-thread-per-env rollouts.
    void push(int player, int env_idx,
              torch::Tensor obs,      // [obs_dim]      CPU or device_
              int64_t action,
              float log_prob,
              float reward,
              float done,
              float value,
              torch::Tensor mask);    // [action_count] CPU or device_

    /// Compute GAE independently per (player, env_idx) trajectory. Trajectories
    /// are truncated at the end of the rollout (no bootstrap past the last
    /// recorded transition).
    void compute_returns(float gamma, float gae_lambda);

    /// Concatenate all valid transitions across both players and all envs.
    /// Returned tensors live on `device_`.
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

    int num_envs()     const { return num_envs_; }
    int num_steps()    const { return num_steps_; }
    int action_count() const { return action_count_; }

private:
    int num_steps_, num_envs_, obs_dim_, action_count_;
    torch::Device device_;

    // Per-player storage. Both players share upper-bound capacity num_steps
    // along the time axis — the actual length of each (p, e) trajectory is
    // counts_[p][e] ≤ num_steps.
    torch::Tensor obs_[2];          // [T, N, obs_dim]
    torch::Tensor actions_[2];      // [T, N]            int64
    torch::Tensor log_probs_[2];    // [T, N]
    torch::Tensor rewards_[2];      // [T, N]
    torch::Tensor dones_[2];        // [T, N]
    torch::Tensor values_[2];       // [T, N]
    torch::Tensor legal_masks_[2];  // [T, N, A]
    torch::Tensor advantages_[2];   // [T, N]
    torch::Tensor returns_[2];      // [T, N]

    std::vector<int32_t> counts_[2];  // [N]
};

// ─────────────────────────────────────────────────────────────────────────────
// PlayerRolloutState — per-env bookkeeping for the per-player buffer.
//
// Each rollout strategy instantiates one PlayerRolloutState per env and drives
// it through record_step / step_reward / flush_on_terminal / flush_on_rollout_end.
// The state tracks, per player:
//   - pending[p]: the most recent transition for p that hasn't had its reward
//     closed out yet (it's closed on p's next action or on termination).
//   - accumulated[p]: the total env reward (in p's frame) received since p's
//     last action.
//   - next_done_flag[p]: the dones value to stamp on p's next recorded
//     transition — flips to 1 after an episode terminates so the first
//     transition of the new episode carries the episode-boundary marker.
// ─────────────────────────────────────────────────────────────────────────────
struct PlayerRolloutState {
    struct Pending {
        bool           has = false;
        torch::Tensor  obs;       // [obs_dim]
        torch::Tensor  mask;      // [action_count]
        int64_t        action   = 0;
        float          log_prob = 0.0f;
        float          value    = 0.0f;
        float          done     = 0.0f;  // dones flag stamped at record time
    };

    Pending pending[2];
    float   accumulated[2]    = {0.0f, 0.0f};
    float   next_done_flag[2] = {0.0f, 0.0f};

    /// Called at each env step by the acting player. Closes out the previous
    /// pending (if any) into the buffer using accumulated[player] as its
    /// reward, then records a new pending for this step.
    void record_step(int player, int env_idx, RolloutBuffer& buf,
                     torch::Tensor obs, torch::Tensor mask,
                     int64_t action, float log_prob, float value) {
        Pending& p = pending[player];
        if (p.has) {
            buf.push(player, env_idx,
                     p.obs, p.action, p.log_prob,
                     accumulated[player], p.done, p.value, p.mask);
            accumulated[player] = 0.0f;
            p.has = false;
        }
        p.has      = true;
        p.obs      = std::move(obs);
        p.mask     = std::move(mask);
        p.action   = action;
        p.log_prob = log_prob;
        p.value    = value;
        p.done     = next_done_flag[player];
        next_done_flag[player] = 0.0f;
    }

    /// Call after env.step() returns reward `env_reward` in the canonical
    /// zero-sum frame (player 0's perspective). Propagates the reward to both
    /// players' accumulators with opposite signs.
    void step_reward(float env_reward) {
        accumulated[0] += env_reward;
        accumulated[1] -= env_reward;
    }

    /// Call on episode termination. Flushes both pending transitions (their
    /// rewards are the terminal accumulators) with done = p.done. Then arms
    /// next_done_flag for both players so their next recorded transition
    /// carries the new-episode boundary.
    void flush_on_terminal(int env_idx, RolloutBuffer& buf) {
        for (int p = 0; p < 2; ++p) {
            Pending& pp = pending[p];
            if (pp.has) {
                buf.push(p, env_idx,
                         pp.obs, pp.action, pp.log_prob,
                         accumulated[p], pp.done, pp.value, pp.mask);
                accumulated[p] = 0.0f;
                pp.has = false;
            }
        }
        next_done_flag[0] = 1.0f;
        next_done_flag[1] = 1.0f;
    }

    /// Call at rollout end to drain any still-pending transitions. These are
    /// flushed with their recorded done flag; compute_returns treats the last
    /// transition of each trajectory as truncated (no bootstrap).
    void flush_on_rollout_end(int env_idx, RolloutBuffer& buf) {
        for (int p = 0; p < 2; ++p) {
            Pending& pp = pending[p];
            if (pp.has) {
                buf.push(p, env_idx,
                         pp.obs, pp.action, pp.log_prob,
                         accumulated[p], pp.done, pp.value, pp.mask);
                accumulated[p] = 0.0f;
                pp.has = false;
            }
        }
    }
};

} // namespace poker_ppo
