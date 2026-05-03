#include "opponent_manager.h"

#include "opponent_pool.h"

#include <algorithm>
#include <random>
#include <unordered_map>

namespace poker_ppo {

OpponentManager::OpponentManager(OpponentPoolConfig  cfg,
                                 int                 obs_dim,
                                 int                 action_count,
                                 int                 hidden_dim,
                                 int                 num_layers,
                                 BetHistoryConfig    hist,
                                 RoundSummaryConfig  round_summary,
                                 torch::Device       device)
    : cfg_(cfg),
      // Derived seed so the manager's episode_rng_ and the pool's internal
      // RNG produce different streams when both are user-seeded.
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
    // Default: pure self-play (no pool override).
    learner_seat_[env_idx] = 0;
    opp_id_[env_idx]       = 0;

    if (!pool_ || pool_->empty()) return;
    if (update_idx < cfg_.warmup_updates) return;

    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (u01(episode_rng_) >= cfg_.p_use_pool) return;

    // HU NLHE is positionally asymmetric (BB / SB), so balance which seat
    // the learner plays to avoid biasing the training distribution.
    std::uniform_int_distribution<int> ui_seat(0, 1);
    learner_seat_[env_idx] = ui_seat(episode_rng_);

    // Restrict to the rollout's pre-sampled IDs when available, so we
    // bound the number of distinct pool snapshots used per rollout (and
    // therefore the number of mini-forwards in apply_action_overrides).
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
    // Pure self-play (op_id == 0) → record both seats so the live network
    // learns from both halves of the experience as before. Pool-active →
    // only the learner seat's transitions go into the buffer.
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

    // Group envs needing override by SnapshotId. Stale IDs (snapshot
    // dropped since the env sampled it) silently fall through to live policy.
    std::unordered_map<uint64_t, std::vector<int>> by_id;
    by_id.reserve(static_cast<size_t>(pool_->size()));
    for (int i = 0; i < N; ++i) {
        const uint64_t id = opp_id_[i];
        if (id == 0) continue;
        if (cp_acc[i] == learner_seat_[i]) continue;  // learner is acting
        if (!pool_->has_id(id))            continue;  // stale — fall through
        by_id[id].push_back(i);
    }
    if (by_id.empty()) return;

    auto a_acc_w  = actions_cpu.accessor<int64_t, 1>();
    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

    for (auto& [id, idxs] : by_id) {
        const int n = static_cast<int>(idxs.size());
        // Build a Long index tensor on CPU, then move to device for
        // index_select. Use a clone so the tensor owns its memory (idxs
        // is a std::vector<int>, not int64).
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
    // Don't fill the pool until warmup completes — early-training snapshots
    // are barely-distinguishable-from-random and dilute the gradient signal
    // once we sample from them.
    if (update_idx < cfg_.warmup_updates) return;
    if (update_idx % cfg_.snapshot_every != 0) return;
    // Reservoir-sampling rejects an offer with probability max_size /
    // seen_count once the pool is full; we don't care about the returned
    // SnapshotId at this site (sampling later goes through sample_id()).
    [[maybe_unused]] const auto _ = pool_->add_snapshot(network);
}

int OpponentManager::size()     const noexcept { return pool_ ? pool_->size()     : 0; }
int OpponentManager::capacity() const noexcept { return pool_ ? pool_->capacity() : 0; }

} // namespace poker_ppo
