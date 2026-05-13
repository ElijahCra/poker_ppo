#pragma once
//
// Frozen past-policy pool + the trainer-side wrapper that drives it to avoid self play cycling
// Keep k past model snapshots, uses reservoir sampling

#include "config.h"
#include "network.h"

#include <torch/torch.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <vector>

namespace poker_ppo {

class OpponentPool {
public:
    using SnapshotId = uint64_t;     // 0 = "no snapshot / use live"

    OpponentPool(int obs_dim, int action_count,
                 int hidden_dim, int num_layers,
                 BetHistoryConfig    hist,
                 RoundSummaryConfig  round_summary,
                 torch::Device       device,
                 int                 max_size,
                 uint64_t            seed = 0,
                 CFVAuxConfig        cfv_aux = {});

    int  size()     const { return static_cast<int>(snapshots_.size()); }
    int  capacity() const { return max_size_; }
    bool empty()    const { return snapshots_.empty(); }

    // Algorithm R: store unconditionally until full, then accept with
    // probability max_size/seen_count and replace a random slot. Returns
    // the new id, or 0 if rejected (or max_size <= 0).
    [[nodiscard]] SnapshotId add_snapshot(const ActorCritic& src);

    // 0 if pool is empty.
    [[nodiscard]] SnapshotId sample_id();

    // Up to n distinct ids without replacement, in random order.
    [[nodiscard]] std::vector<SnapshotId> sample_ids(int n);

    [[nodiscard]] bool has_id(SnapshotId id) const;

    // obs/legal_mask must be on device_. Caller must ensure has_id(id)
    // (TORCH_CHECK otherwise). Returns int64 CPU actions.
    torch::Tensor select_actions(SnapshotId id,
                                 const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask);

private:
    struct Entry { SnapshotId id; ActorCritic net; };

    int obs_dim_, action_count_, hidden_dim_, num_layers_;
    BetHistoryConfig    hist_;
    RoundSummaryConfig  round_summary_;
    CFVAuxConfig        cfv_aux_;
    torch::Device       device_;
    int                 max_size_;

    std::deque<Entry>   snapshots_;
    SnapshotId          next_id_     = 1;   // 0 reserved
    uint64_t            seen_count_  = 0;
    std::mt19937        rng_;
};

// Trainer-side state. All entry points are no-ops when the pool is
// disabled or empty (warmup).
class OpponentManager {
public:
    OpponentManager(OpponentPoolConfig  cfg,
                    int                 obs_dim,
                    int                 action_count,
                    int                 hidden_dim,
                    int                 num_layers,
                    BetHistoryConfig    hist,
                    RoundSummaryConfig  round_summary,
                    torch::Device       device,
                    CFVAuxConfig        cfv_aux = {});

    ~OpponentManager();

    void reset_assignments(int num_envs);

    // Sample the rollout's eligible ids. Capped at max_unique_per_rollout
    // so apply_action_overrides runs at most that many forwards per step.
    void prepare_rollout(int update_idx);

    // Re-roll learner_seat / opp_id for env_idx after its hand terminates.
    void on_episode_terminal(int env_idx, int update_idx);

    // Pure self-play (op_id == 0): always true. Pool-active: only the
    // learner seat's transitions are recorded.
    [[nodiscard]] bool should_record(int env_idx,
                                     int acting_player) const noexcept;

    // Overwrite slots in actions_cpu where a pool snapshot is acting.
    void apply_action_overrides(const torch::Tensor& cur_obs,
                                const torch::Tensor& cur_mask,
                                const torch::Tensor& cur_player_cpu,
                                torch::Tensor&       actions_cpu);

    // Append network when (a) enabled, (b) past warmup, (c) update_idx
    // hits snapshot_every.
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
