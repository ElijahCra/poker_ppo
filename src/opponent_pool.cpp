#include "opponent_pool.h"

#include <algorithm>
#include <numeric>

namespace poker_ppo {

OpponentPool::OpponentPool(int obs_dim, int action_count,
                           int hidden_dim, int num_layers,
                           BetHistoryConfig    hist,
                           RoundSummaryConfig  round_summary,
                           torch::Device       device,
                           int                 max_size,
                           uint64_t            seed)
    : obs_dim_(obs_dim), action_count_(action_count),
      hidden_dim_(hidden_dim), num_layers_(num_layers),
      hist_(hist), round_summary_(round_summary),
      device_(device), max_size_(std::max(0, max_size)),
      rng_(seed ? seed : std::random_device{}()) {}

OpponentPool::SnapshotId OpponentPool::add_snapshot(const ActorCritic& src) {
    if (max_size_ <= 0) return 0;

    // Vitter's Algorithm R. seen_count_ is incremented on every offer (whether
    // accepted or not) and is the denominator in the inclusion probability,
    // giving each historical offer equal odds max_size_/seen_count_ of
    // currently being in the pool.
    ++seen_count_;

    // Pool not yet full — accept unconditionally.
    if (static_cast<int>(snapshots_.size()) < max_size_) {
        const SnapshotId id = next_id_++;
        snapshots_.push_back({id, clone_actor_critic(src, obs_dim_, action_count_,
                                        hidden_dim_, num_layers_,
                                        hist_, round_summary_, device_)});
        return id;
    }

    // Pool full — accept with probability max_size_/seen_count_.
    std::uniform_int_distribution<uint64_t> ui(1, seen_count_);
    if (ui(rng_) > static_cast<uint64_t>(max_size_)) {
        return 0;  // rejected; offered snapshot is discarded without cloning.
    }

    // Accept: replace a uniformly-random existing slot. The replaced entry's
    // ID is retired — any env still holding it falls through to live policy
    // via has_id() returning false.
    std::uniform_int_distribution<int> us(0, max_size_ - 1);
    const int slot = us(rng_);
    const SnapshotId id = next_id_++;
    snapshots_[slot] = {id, clone_actor_critic(src, obs_dim_, action_count_,
                                        hidden_dim_, num_layers_,
                                        hist_, round_summary_, device_)};
    return id;
}

OpponentPool::SnapshotId OpponentPool::sample_id() {
    if (snapshots_.empty()) return 0;
    std::uniform_int_distribution<int> ui(
        0, static_cast<int>(snapshots_.size()) - 1);
    return snapshots_[ui(rng_)].id;
}

std::vector<OpponentPool::SnapshotId> OpponentPool::sample_ids(int n) {
    std::vector<SnapshotId> out;
    if (snapshots_.empty() || n <= 0) return out;

    const int K = static_cast<int>(snapshots_.size());
    n = std::min(n, K);
    std::vector<int> indices(K);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng_);

    out.reserve(n);
    for (int i = 0; i < n; ++i) {
        out.push_back(snapshots_[indices[i]].id);
    }
    return out;
}

bool OpponentPool::has_id(SnapshotId id) const {
    if (id == 0) return false;
    for (const auto& e : snapshots_) {
        if (e.id == id) return true;
    }
    return false;
}

torch::Tensor OpponentPool::select_actions(SnapshotId id,
                                           const torch::Tensor& obs,
                                           const torch::Tensor& legal_mask) {
    torch::NoGradGuard ng;
    for (auto& e : snapshots_) {
        if (e.id == id) {
            auto ar = e.net->get_action(obs, legal_mask);
            return ar.action.to(torch::kCPU).contiguous();
        }
    }
    TORCH_CHECK(false, "OpponentPool::select_actions: id not in pool");
}

} // namespace poker_ppo
