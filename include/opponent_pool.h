#pragma once
//
// opponent_pool.h — frozen past-policy pool for self-play stabilisation.
//
// Pure self-play (current vs current) cycles: the policy collapses to exploit
// its current self, then diverges into a different collapsed mode, etc. The
// signature is entropy oscillation, return-variance swings, and regression
// against fixed anchors mid-training.
//
// Standard fix: maintain a FIFO pool of K past snapshots. Each rollout, with
// probability `p_use_pool`, an env's opponent seat plays a uniformly-sampled
// pool snapshot instead of the live policy. The live seat's transitions are
// the only ones the buffer records — opponent transitions are never learned
// from. This forces the learner to be robust against past versions of itself.
//
// Snapshots are deep-copied at add_snapshot() time so subsequent training
// updates can't mutate them. Inference is always under NoGradGuard.
//
// Snapshots are addressed by a monotonically-increasing SnapshotId rather
// than by deque index. When the pool drops an old snapshot, its ID is simply
// retired — any env that still holds a stale ID falls through to the live
// policy via has_id() returning false. This avoids "opponent silently swaps
// mid-episode" if a drop happens between when an env sampled its opponent
// and when its episode ends.
//

#include "network.h"
#include "types.h"

#include <torch/torch.h>

#include <cstdint>
#include <deque>
#include <random>

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

    /// Snapshot `src` into the pool. FIFO: drops the oldest snapshot when
    /// full (its ID stays retired). Returns the new SnapshotId, or 0 if
    /// max_size <= 0.
    SnapshotId add_snapshot(const ActorCritic& src);

    /// Sample one of the live snapshots' IDs uniformly. Returns 0 if empty.
    SnapshotId sample_id();

    /// Is `id` currently in the pool?
    bool has_id(SnapshotId id) const;

    /// Run inference for snapshot `id` over a sub-batch.
    /// `obs`/`legal_mask` must already live on `device_`. Caller must ensure
    /// `has_id(id)` — fails a TORCH_CHECK otherwise. Returns int64 CPU
    /// actions of the same length as the sub-batch.
    torch::Tensor select_actions(SnapshotId id,
                                 const torch::Tensor& obs,
                                 const torch::Tensor& legal_mask);

private:
    ActorCritic clone_network(const ActorCritic& src);

    struct Entry { SnapshotId id; ActorCritic net; };

    int obs_dim_, action_count_, hidden_dim_, num_layers_;
    BetHistoryConfig    hist_;
    RoundSummaryConfig  round_summary_;
    torch::Device       device_;
    int                 max_size_;

    std::deque<Entry>   snapshots_;
    SnapshotId          next_id_ = 1;   // 0 reserved
    std::mt19937        rng_;
};

} // namespace poker_ppo
