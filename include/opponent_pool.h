#pragma once
//
// opponent_pool.h — frozen past-policy pool for self-play stabilisation.
//
// Pure self-play (current vs current) cycles: the policy collapses to
// exploit its current self, diverges into a different collapsed mode,
// etc. Mitigation: keep a reservoir-sampled pool of K past snapshots and,
// with probability `p_use_pool`, play an env's opponent seat from a
// uniformly-sampled pool member. Only live-seat transitions are recorded.
//
// Reservoir sampling (Vitter's Algorithm R) gives every snapshot ever
// offered an equal probability max_size/seen_count of being in the pool.
// Unlike FIFO, easy post-warmup snapshots aren't evicted as the pool
// fills, so the pool's effective difficulty doesn't rise monotonically
// and the learner doesn't over-fold-collapse against rising opponents.
//
// Snapshots are addressed by a monotonically-increasing SnapshotId, not
// container index — when a snapshot is replaced its ID is retired, and
// any env still holding the stale ID falls through to the live policy
// via has_id() returning false (no mid-episode opponent swaps).
//

#include "config.h"
#include "network.h"

#include <torch/torch.h>

#include <cstdint>
#include <deque>
#include <random>
#include <vector>

namespace poker_ppo {

class OpponentPool {
public:
    using SnapshotId = uint64_t;     // 0 sentinel = "no snapshot / use live"

    OpponentPool(int obs_dim, int action_count,
                 int hidden_dim, int num_layers,
                 BetHistoryConfig    hist,
                 RoundSummaryConfig  round_summary,
                 torch::Device       device,
                 int                 max_size,
                 uint64_t            seed = 0);

    int  size()     const { return static_cast<int>(snapshots_.size()); }
    int  capacity() const { return max_size_; }
    bool empty()    const { return snapshots_.empty(); }

    /// Offer `src` to the pool. Until the reservoir is full we always store;
    /// once full we accept with probability max_size/seen_count and replace a
    /// uniformly random existing slot — Vitter's Algorithm R. Returns the new
    /// SnapshotId on store, or 0 if the offer was rejected (or max_size <= 0).
    [[nodiscard]] SnapshotId add_snapshot(const ActorCritic& src);

    /// Sample one of the live snapshots' IDs uniformly. Returns 0 if empty.
    [[nodiscard]] SnapshotId sample_id();

    /// Sample up to `n` distinct snapshot IDs without replacement. Returns at
    /// most `min(n, size())` IDs in random order; empty vector if pool is empty.
    [[nodiscard]] std::vector<SnapshotId> sample_ids(int n);

    /// Is `id` currently in the pool?
    [[nodiscard]] bool has_id(SnapshotId id) const;

    /// Run inference for snapshot `id` over a sub-batch.
    /// `obs`/`legal_mask` must already live on `device_`. Caller must ensure
    /// `has_id(id)` — fails a TORCH_CHECK otherwise. Returns int64 CPU
    /// actions of the same length as the sub-batch.
    torch::Tensor select_actions(SnapshotId id,
                                 const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask);

private:
    struct Entry { SnapshotId id; ActorCritic net; };

    int obs_dim_, action_count_, hidden_dim_, num_layers_;
    BetHistoryConfig    hist_;
    RoundSummaryConfig  round_summary_;
    torch::Device       device_;
    int                 max_size_;

    std::deque<Entry>   snapshots_;
    SnapshotId          next_id_     = 1;   // 0 reserved
    uint64_t            seen_count_  = 0;   // total snapshots ever offered
    std::mt19937        rng_;
};

} // namespace poker_ppo
