#pragma once
//
// opponent_manager.h — opponent-pool state used by the trainer during a
// rollout. Entry points: prepare_rollout, apply_action_overrides,
// on_episode_terminal, maybe_snapshot. All no-op when the pool is
// disabled or empty (warmup). The reservoir + snapshot inference lives
// in opponent_pool.h.
//

#include "config.h"
#include "network.h"

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace poker_ppo {

class OpponentPool;  // fwd-decl; full type in opponent_pool.h

class OpponentManager {
public:
    OpponentManager(OpponentPoolConfig  cfg,
                    int                 obs_dim,
                    int                 action_count,
                    int                 hidden_dim,
                    int                 num_layers,
                    BetHistoryConfig    hist,
                    RoundSummaryConfig  round_summary,
                    torch::Device       device);

    // Out-of-line so callers don't need OpponentPool's full definition.
    ~OpponentManager();

    // Resize per-env tracking + reset everyone to "self-play".
    void reset_assignments(int num_envs);

    // Sample the rollout's eligible pool IDs. Capped at
    // cfg.max_unique_per_rollout so apply_action_overrides() runs at most
    // that many forward passes per step.
    void prepare_rollout(int update_idx);

    // Re-roll learner_seat / opp_id for env `env_idx`. Called when that
    // env's hand just terminated.
    void on_episode_terminal(int env_idx, int update_idx);

    // Whether transitions for (env_idx, acting_player) belong in the
    // rollout buffer. Pure self-play: always true. Pool-active: only the
    // learner seat's transitions are recorded.
    [[nodiscard]] bool should_record(int env_idx,
                                     int acting_player) const noexcept;

    // Overwrite slots in `actions_cpu` where a pool snapshot is acting.
    // No-op when pool is empty or no env has a pool override active.
    void apply_action_overrides(const torch::Tensor& cur_obs,
                                const torch::Tensor& cur_mask,
                                const torch::Tensor& cur_player_cpu,
                                torch::Tensor&       actions_cpu);

    // Append `network` to the pool when (a) enabled, (b) past warmup,
    // (c) update_idx aligns with snapshot_every. Otherwise no-op.
    void maybe_snapshot(int update_idx, const ActorCritic& network);

    [[nodiscard]] int  size()     const noexcept;
    [[nodiscard]] int  capacity() const noexcept;
    [[nodiscard]] bool enabled()  const noexcept { return pool_ != nullptr; }

private:
    OpponentPoolConfig            cfg_;
    std::unique_ptr<OpponentPool> pool_;
    std::vector<int>              learner_seat_;
    std::vector<uint64_t>         opp_id_;
    std::vector<uint64_t>         rollout_pool_ids_;
    std::mt19937                  episode_rng_;
};

} // namespace poker_ppo
