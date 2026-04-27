#include "opponent_pool.h"

#include <algorithm>

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

ActorCritic OpponentPool::clone_network(const ActorCritic& src) {
    // Same shape as the trainer's network — must stay in sync if the trainer's
    // architecture flags ever drift from ppo_cfg.
    ActorCritic dst(obs_dim_, action_count_, hidden_dim_, num_layers_,
                    hist_, round_summary_);
    dst->to(device_);

    torch::NoGradGuard ng;
    auto sp = src->parameters();
    auto dp = dst->parameters();
    TORCH_CHECK(sp.size() == dp.size(),
                "OpponentPool::clone_network: parameter count mismatch");
    for (size_t i = 0; i < sp.size(); ++i) {
        dp[i].copy_(sp[i].detach().to(device_));
    }
    auto sb = src->buffers();
    auto db = dst->buffers();
    TORCH_CHECK(sb.size() == db.size(),
                "OpponentPool::clone_network: buffer count mismatch");
    for (size_t i = 0; i < sb.size(); ++i) {
        db[i].copy_(sb[i].detach().to(device_));
    }
    dst->eval();
    return dst;
}

OpponentPool::SnapshotId OpponentPool::add_snapshot(const ActorCritic& src) {
    if (max_size_ <= 0) return 0;
    while (static_cast<int>(snapshots_.size()) >= max_size_) {
        snapshots_.pop_front();
    }
    const SnapshotId id = next_id_++;
    snapshots_.push_back({id, clone_network(src)});
    return id;
}

OpponentPool::SnapshotId OpponentPool::sample_id() {
    if (snapshots_.empty()) return 0;
    std::uniform_int_distribution<int> ui(
        0, static_cast<int>(snapshots_.size()) - 1);
    return snapshots_[ui(rng_)].id;
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
