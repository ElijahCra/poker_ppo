#include "opponent_manager.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_map>

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

    // Algorithm R: seen_count_ is the inclusion-probability denominator.
    // Increment on every offer (accepted or not).
    ++seen_count_;

    if (static_cast<int>(snapshots_.size()) < max_size_) {
        const SnapshotId id = next_id_++;
        snapshots_.push_back({id, clone_actor_critic(src, obs_dim_, action_count_,
                                        hidden_dim_, num_layers_,
                                        hist_, round_summary_, device_)});
        return id;
    }

    // Accept with probability max_size/seen_count.
    std::uniform_int_distribution<uint64_t> ui(1, seen_count_);
    if (ui(rng_) > static_cast<uint64_t>(max_size_)) {
        return 0;
    }

    // Replace a random slot; the replaced ID retires (envs holding it
    // fall through via has_id()).
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

OpponentManager::OpponentManager(OpponentPoolConfig  cfg,
                                 int                 obs_dim,
                                 int                 action_count,
                                 int                 hidden_dim,
                                 int                 num_layers,
                                 BetHistoryConfig    hist,
                                 RoundSummaryConfig  round_summary,
                                 torch::Device       device)
    : cfg_(cfg),
      // Different stream from the pool's RNG when both are user-seeded.
      episode_rng_(cfg.seed ? cfg.seed
                            : std::random_device{}())
{
    if (!cfg_.enabled) return;

    const uint64_t pool_seed = cfg_.seed
        ? cfg_.seed ^ 0x9E3779B97F4A7C15ULL
        : 0;
    pool_ = std::make_unique<OpponentPool>(
        obs_dim, action_count, hidden_dim, num_layers,
        hist, round_summary, device, cfg_.max_size, pool_seed);
}

OpponentManager::~OpponentManager() = default;

void OpponentManager::reset_assignments(int num_envs) {
    learner_seat_.assign(num_envs, 0);
    opp_id_.assign(num_envs, 0);
}

void OpponentManager::prepare_rollout(int update_idx) {
    rollout_pool_ids_.clear();
    if (!pool_ || pool_->empty()) return;
    if (update_idx < cfg_.warmup_updates) return;

    const int n = std::max(1, cfg_.max_unique_per_rollout);
    rollout_pool_ids_ = pool_->sample_ids(n);
}

void OpponentManager::on_episode_terminal(int env_idx, int update_idx) {
    // Default: pure self-play.
    learner_seat_[env_idx] = 0;
    opp_id_[env_idx]       = 0;

    if (!pool_ || pool_->empty()) return;
    if (update_idx < cfg_.warmup_updates) return;

    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (u01(episode_rng_) >= cfg_.p_use_pool) return;

    // HU NLHE is positionally asymmetric — balance which seat the learner
    // plays so we don't bias the training distribution.
    std::uniform_int_distribution<int> ui_seat(0, 1);
    learner_seat_[env_idx] = ui_seat(episode_rng_);

    // Bound distinct snapshots used per rollout (= mini-forwards in
    // apply_action_overrides) by drawing from the pre-sampled set.
    if (!rollout_pool_ids_.empty()) {
        std::uniform_int_distribution<size_t> ui(
            0, rollout_pool_ids_.size() - 1);
        opp_id_[env_idx] = rollout_pool_ids_[ui(episode_rng_)];
    } else {
        opp_id_[env_idx] = pool_->sample_id();
    }
}

bool OpponentManager::should_record(int env_idx,
                                    int acting_player) const noexcept {
    // Pure self-play: record both seats. Pool-active: learner only.
    const uint64_t op_id = opp_id_[env_idx];
    return (op_id == 0) || (acting_player == learner_seat_[env_idx]);
}

void OpponentManager::apply_action_overrides(
    const torch::Tensor& cur_obs,
    const torch::Tensor& cur_mask,
    const torch::Tensor& cur_player_cpu,
    torch::Tensor&       actions_cpu)
{
    if (!pool_ || pool_->empty()) return;

    const int N = static_cast<int>(actions_cpu.size(0));
    auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();

    // Group by SnapshotId. Stale IDs fall through to live policy.
    std::unordered_map<uint64_t, std::vector<int>> by_id;
    by_id.reserve(static_cast<size_t>(pool_->size()));
    for (int i = 0; i < N; ++i) {
        const uint64_t id = opp_id_[i];
        if (id == 0) continue;
        if (cp_acc[i] == learner_seat_[i]) continue;  // learner is acting
        if (!pool_->has_id(id))            continue;  // stale
        by_id[id].push_back(i);
    }
    if (by_id.empty()) return;

    auto a_acc_w  = actions_cpu.accessor<int64_t, 1>();
    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

    for (auto& [id, idxs] : by_id) {
        const int n = static_cast<int>(idxs.size());
        // idxs is int, not int64 — clone so the tensor owns its memory.
        std::vector<int64_t> idx64(idxs.begin(), idxs.end());
        auto idx_cpu = torch::from_blob(idx64.data(), {n}, idx_opts).clone();
        auto idx_dev = idx_cpu.to(cur_obs.device());

        auto sub_obs  = cur_obs.index_select(0, idx_dev);
        auto sub_mask = cur_mask.index_select(0, idx_dev);

        auto pool_actions = pool_->select_actions(id, sub_obs, sub_mask);
        auto pa = pool_actions.accessor<int64_t, 1>();
        for (int k = 0; k < n; ++k) {
            a_acc_w[idxs[k]] = pa[k];
        }
    }
}

void OpponentManager::maybe_snapshot(int update_idx,
                                     const ActorCritic& network) {
    if (!pool_) return;
    if (cfg_.snapshot_every <= 0) return;
    if (update_idx <= 0) return;
    // Pre-warmup snapshots are near-random and dilute the gradient signal
    // once we start sampling from them.
    if (update_idx < cfg_.warmup_updates) return;
    if (update_idx % cfg_.snapshot_every != 0) return;
    [[maybe_unused]] const auto _ = pool_->add_snapshot(network);
}

int OpponentManager::size()     const noexcept { return pool_ ? pool_->size()     : 0; }
int OpponentManager::capacity() const noexcept { return pool_ ? pool_->capacity() : 0; }

} // namespace poker_ppo
