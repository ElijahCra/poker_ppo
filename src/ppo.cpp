#include "ppo.h"
#include "opponent_pool.h"
#include "rollout_pool.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>
#include <unordered_map>

namespace poker_ppo {

// ═════════════════════════════════════════════════════════════════════════════
// VectorizedEnv
// ═════════════════════════════════════════════════════════════════════════════

VectorizedEnv::VectorizedEnv(IPokerEnvironmentFactory& factory,
                             const BetConfig& cfg, int num_envs) {
    envs_.reserve(num_envs);
    for (int i = 0; i < num_envs; ++i)
        envs_.push_back(factory.create(cfg));

    int N = num_envs, D = obs_dim(), A = action_count();
    obs_buf_     = torch::zeros({N, D});
    rewards_buf_ = torch::zeros({N});
    dones_buf_   = torch::zeros({N});
    masks_buf_   = torch::zeros({N, A});
    players_buf_ = torch::zeros({N}, torch::kInt32);
}

torch::Tensor VectorizedEnv::reset_all() {
    int N = num_envs();
    int D = obs_dim();
    auto obs = torch::zeros({N, D});
    for (int i = 0; i < N; ++i) {
        auto result = envs_[i]->reset();
        obs[i] = result.observation;
    }
    return obs;
}


// ─────────────────────────────────────────────────────────────────────────────
// build_bootstrap — assemble the [2, N] bootstrap_values / bootstrap_terminal
// tensors that compute_returns expects. Forwards `cur_obs` through the critic
// to get V at the rollout-end state (from the next-acting player's view), and
// uses zero-sum negation to derive V from the other player's view.
// ─────────────────────────────────────────────────────────────────────────────
namespace {

struct BootstrapTensors {
    torch::Tensor values;     // [2, N] float32, CPU
    torch::Tensor terminal;   // [2, N] float32, CPU
};

BootstrapTensors build_bootstrap(
    ActorCritic& network,
    const torch::Tensor& cur_obs,           // [N, obs_dim] on device
    const torch::Tensor& cur_player_cpu,    // [N] int32 on CPU
    const std::vector<PlayerRolloutState>& player_state) {

    const int N = static_cast<int>(cur_obs.size(0));
    auto f_cpu = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    // Critic-only forward for V at carry obs. NoGrad is assumed in scope at
    // the call site (rollout collectors run under NoGradGuard).
    auto V_carry = network->get_value(cur_obs).detach().to(torch::kCPU).contiguous();

    auto values   = torch::zeros({2, N}, f_cpu);
    auto terminal = torch::zeros({2, N}, f_cpu);

    auto v_acc  = V_carry.accessor<float, 1>();
    auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();
    auto bv_acc = values.accessor<float, 2>();
    auto bt_acc = terminal.accessor<float, 2>();

    for (int i = 0; i < N; ++i) {
        const int next_actor = cp_acc[i];
        const float v        = v_acc[i];
        // V at carry obs is from the next-acting player's perspective; the
        // other player's V is its zero-sum negation.
        bv_acc[next_actor][i]     = v;
        bv_acc[1 - next_actor][i] = -v;
        bt_acc[0][i] = player_state[i].tail_was_terminal[0] ? 1.0f : 0.0f;
        bt_acc[1][i] = player_state[i].tail_was_terminal[1] ? 1.0f : 0.0f;
    }
    return {values, terminal};
}

} // namespace

// ═════════════════════════════════════════════════════════════════════════════
// PPOTrainer
// ═════════════════════════════════════════════════════════════════════════════

PPOTrainer::PPOTrainer(IPokerEnvironmentFactory& env_factory,
                       const BetConfig& bet_cfg,
                       const PPOConfig& ppo_cfg,
                       torch::Device device)
    : cfg_(ppo_cfg), bet_cfg_(bet_cfg), device_(device),
      network_(nullptr),
      episode_rng_(cfg_.opp_pool.seed
                   ? cfg_.opp_pool.seed
                   : std::random_device{}())
{
    // Create vectorized environment
    vec_env_ = std::make_unique<VectorizedEnv>(
        env_factory, bet_cfg, cfg_.num_envs);

    int obs_dim      = vec_env_->obs_dim();
    int action_count = vec_env_->action_count();

    // Create network
    network_ = ActorCritic(obs_dim, action_count,
                           cfg_.hidden_dim, cfg_.num_layers,
                           cfg_.hist, cfg_.round_summary);
    network_->to(device_);

    // Optimiser
    optimizer_ = std::make_unique<torch::optim::Adam>(
        network_->parameters(),
        torch::optim::AdamOptions(cfg_.learning_rate));

    // Rollout buffer (lives on device — avoids CPU round-trip each rollout)
    buffer_ = std::make_unique<RolloutBuffer>(
        cfg_.num_steps, cfg_.num_envs, obs_dim, action_count, device_);

    // Opponent pool: derived seed so its RNG and the trainer's episode_rng_
    // don't produce the same stream when both are user-seeded.
    if (cfg_.opp_pool.enabled) {
        const uint64_t pool_seed = cfg_.opp_pool.seed
            ? cfg_.opp_pool.seed ^ 0x9E3779B97F4A7C15ULL
            : 0;
        opp_pool_ = std::make_unique<OpponentPool>(
            obs_dim, action_count,
            cfg_.hidden_dim, cfg_.num_layers,
            cfg_.hist, cfg_.round_summary,
            device_, cfg_.opp_pool.max_size, pool_seed);
    }

    learner_seat_.assign(cfg_.num_envs, 0);
    opp_id_.assign(cfg_.num_envs, 0);
}

PPOTrainer::~PPOTrainer() = default;

int PPOTrainer::opponent_pool_size() const {
    return opp_pool_ ? opp_pool_->size() : 0;
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::init_carry_state() {
    int D = vec_env_->obs_dim();
    int A = vec_env_->action_count();
    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    carry_obs_            = torch::zeros({cfg_.num_envs, D}, f_cpu);
    carry_legal_mask_     = torch::zeros({cfg_.num_envs, A}, f_cpu);
    carry_current_player_ = torch::zeros({cfg_.num_envs}, i32_cpu);
    carry_done_           = torch::zeros({cfg_.num_envs}, f_cpu);

    vec_env_->reset_all();
    for (int i = 0; i < cfg_.num_envs; ++i) {
        carry_obs_[i]        = vec_env_->env(i).observation();
        carry_legal_mask_[i] = vec_env_->env(i).legal_action_mask();
        carry_current_player_.accessor<int32_t, 1>()[i] =
            static_cast<int32_t>(vec_env_->env(i).current_player());
    }

    // Per-env opponent assignment. Both vectors stay all-zero until pool
    // warmup; first roll happens at the first episode boundary inside a
    // rollout (or here, for envs that skip warmup).
    learner_seat_.assign(cfg_.num_envs, 0);
    opp_id_.assign(cfg_.num_envs, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Opponent-pool helpers — all no-op when the pool is disabled.
// ─────────────────────────────────────────────────────────────────────────────

void PPOTrainer::maybe_snapshot() {
    if (!opp_pool_) return;
    const auto& pcfg = cfg_.opp_pool;
    if (pcfg.snapshot_every <= 0) return;
    if (update_idx_ <= 0) return;
    // Don't start filling the pool until warmup completes — early-training
    // snapshots are barely-distinguishable-from-random and dilute the
    // gradient signal once we sample from them.
    if (update_idx_ < pcfg.warmup_updates) return;
    if (update_idx_ % pcfg.snapshot_every != 0) return;
    opp_pool_->add_snapshot(network_);
}

void PPOTrainer::prepare_rollout_pool_ids() {
    rollout_pool_ids_.clear();
    if (!opp_pool_ || opp_pool_->empty()) return;
    if (update_idx_ < cfg_.opp_pool.warmup_updates) return;

    const int n = std::max(1, cfg_.opp_pool.max_unique_per_rollout);
    rollout_pool_ids_ = opp_pool_->sample_ids(n);
}

void PPOTrainer::roll_episode_assignment(int env_idx) {
    // Default: pure self-play (no pool override).
    learner_seat_[env_idx] = 0;
    opp_id_[env_idx]       = 0;

    if (!opp_pool_ || opp_pool_->empty()) return;
    if (update_idx_ < cfg_.opp_pool.warmup_updates) return;

    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    if (u01(episode_rng_) >= cfg_.opp_pool.p_use_pool) return;

    // Heads-up NLHE is positionally asymmetric (BB / SB), so balance which
    // seat the learner plays to avoid biasing the training distribution.
    std::uniform_int_distribution<int> ui_seat(0, 1);
    learner_seat_[env_idx] = ui_seat(episode_rng_);

    // Restrict to the rollout's pre-sampled pool IDs when available, so we
    // bound the number of distinct pool snapshots used per rollout (and
    // therefore the number of mini-forwards in apply_pool_overrides).
    if (!rollout_pool_ids_.empty()) {
        std::uniform_int_distribution<size_t> ui(
            0, rollout_pool_ids_.size() - 1);
        opp_id_[env_idx] = rollout_pool_ids_[ui(episode_rng_)];
    } else {
        opp_id_[env_idx] = opp_pool_->sample_id();
    }
}

void PPOTrainer::apply_pool_overrides(
    const torch::Tensor& cur_obs,
    const torch::Tensor& cur_mask,
    const torch::Tensor& cur_player_cpu,
    torch::Tensor& actions_cpu) {
    if (!opp_pool_ || opp_pool_->empty()) return;

    const int N = static_cast<int>(actions_cpu.size(0));
    auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();

    // Group envs needing override by SnapshotId. Stale IDs (snapshot dropped
    // since the env sampled it) silently fall through to live policy.
    std::unordered_map<uint64_t, std::vector<int>> by_id;
    by_id.reserve(static_cast<size_t>(opp_pool_->size()));
    for (int i = 0; i < N; ++i) {
        const uint64_t id = opp_id_[i];
        if (id == 0) continue;
        if (cp_acc[i] == learner_seat_[i]) continue;  // learner is acting
        if (!opp_pool_->has_id(id))         continue;  // stale — fall through
        by_id[id].push_back(i);
    }
    if (by_id.empty()) return;

    auto a_acc_w = actions_cpu.accessor<int64_t, 1>();
    auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

    for (auto& [id, idxs] : by_id) {
        const int n = static_cast<int>(idxs.size());
        // Build a Long index tensor on CPU, then move to device for index_select.
        // Use a separate buffer (clone) so the tensor owns its memory.
        std::vector<int64_t> idx64(idxs.begin(), idxs.end());
        auto idx_cpu = torch::from_blob(idx64.data(), {n}, idx_opts).clone();
        auto idx_dev = idx_cpu.to(cur_obs.device());

        auto sub_obs  = cur_obs.index_select(0, idx_dev);
        auto sub_mask = cur_mask.index_select(0, idx_dev);

        auto pool_actions = opp_pool_->select_actions(id, sub_obs, sub_mask);
        auto pa = pool_actions.accessor<int64_t, 1>();
        for (int k = 0; k < n; ++k) {
            a_acc_w[idxs[k]] = pa[k];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::train() {
    init_carry_state();

    // Default rollout strategy if the caller didn't pick one.
    if (!rollout_fn_) {
        rollout_fn_ = [](PPOTrainer& t) { t.collect_rollout_threadpool(); };
    }

    int total_updates = cfg_.num_updates();

    for (update_idx_ = 0; update_idx_ < total_updates; ++update_idx_) {
        // Anneal learning rate. Linear from `learning_rate` at update 0 down
        // to `learning_rate * min_lr_frac` at the final update — the floor
        // avoids the dead-final-quarter where lr ≈ 0 and updates stop moving
        // the policy.
        if (cfg_.anneal_lr) {
            const float frac = 1.0f - static_cast<float>(update_idx_) / total_updates;
            const float floor_frac = std::max(0.0f, cfg_.min_lr_frac);
            const float lr = cfg_.learning_rate * std::max(frac, floor_frac);
            for (auto& pg : optimizer_->param_groups())
                static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr);
        }

        using clock = std::chrono::steady_clock;
        using ms    = std::chrono::duration<double, std::milli>;

        auto t0 = clock::now();
        at::set_num_threads(1);
        rollout_fn_(*this);
        auto t1 = clock::now();
        at::set_num_threads(std::thread::hardware_concurrency());
        auto stats = update();
        auto t2 = clock::now();

        stats.rollout_ms = ms(t1 - t0).count();
        stats.update_ms  = ms(t2 - t1).count();

        if (log_cb_) log_cb_(stats);

        if (update_idx_ % 10 == 0) {
            std::cout << "[update " << update_idx_ << "]"
                      << "  rollout=" << stats.rollout_ms << "ms"
                      << "  update=" << stats.update_ms << "ms";
            if (opp_pool_) {
                std::cout << "  pool=" << opp_pool_->size()
                          << "/" << opp_pool_->capacity();
            }
            std::cout << "\n";
        }

        // Snapshot the post-update weights into the opponent pool. No-op
        // when disabled or off the snapshot cadence.
        maybe_snapshot();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Serial rollout — single-thread loop with per-player buffer accumulation.
// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::collect_rollout_serial() {
    network_->eval();
    torch::NoGradGuard no_grad;

    buffer_->clear();
    prepare_rollout_pool_ids();

    auto& envs = vec_env_->envs_mut();
    const int N = cfg_.num_envs;
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();

    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    // Per-env per-player rollout state. Seeded from carry_done_ so the first
    // transition after a previous-rollout reset carries the new-episode marker.
    std::vector<PlayerRolloutState> player_state(N);
    for (int i = 0; i < N; ++i) {
        const float d = carry_done_.accessor<float, 1>()[i];
        player_state[i].next_done_flag[0] = d;
        player_state[i].next_done_flag[1] = d;
    }

    // Pull carry state onto device for the forward pass.
    auto cur_obs    = carry_obs_.to(device_);
    auto cur_mask   = carry_legal_mask_.to(device_);
    auto cur_player = carry_current_player_.to(device_);

    for (int step = 0; step < cfg_.num_steps; ++step) {
        auto ar = network_->get_action(cur_obs, cur_mask);

        auto actions_cpu    = ar.action.to(torch::kCPU).contiguous();
        auto logp_cpu       = ar.log_prob.to(torch::kCPU).contiguous();
        auto value_cpu      = ar.value.to(torch::kCPU).contiguous();
        auto cur_obs_cpu    = cur_obs.to(torch::kCPU).contiguous();
        auto cur_mask_cpu   = cur_mask.to(torch::kCPU).contiguous();
        auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();

        // Override actions for envs where a pool snapshot is acting as the
        // non-learner seat. No-op when the pool is disabled or empty.
        apply_pool_overrides(cur_obs, cur_mask, cur_player_cpu, actions_cpu);

        auto a_acc  = actions_cpu.accessor<int64_t, 1>();
        auto lp_acc = logp_cpu.accessor<float, 1>();
        auto v_acc  = value_cpu.accessor<float, 1>();
        auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();

        auto next_obs    = torch::zeros({N, D}, f_cpu);
        auto next_mask   = torch::zeros({N, A}, f_cpu);
        auto next_player = torch::zeros({N},    i32_cpu);
        auto np_acc      = next_player.accessor<int32_t, 1>();

        std::vector<uint8_t> did_reset(N, 0);

        for (int i = 0; i < N; ++i) {
            const int      acting = cp_acc[i];
            const uint64_t op_id  = opp_id_[i];
            // When pool is active for this env, only the learner seat's
            // transitions go into the buffer. Pure self-play (op_id == 0)
            // keeps recording both seats so the live network learns from
            // both halves of the experience as before.
            const bool record = (op_id == 0) || (acting == learner_seat_[i]);

            if (record) {
                player_state[i].record_step(
                    acting, i, *buffer_,
                    cur_obs_cpu[i], cur_mask_cpu[i],
                    a_acc[i], lp_acc[i], v_acc[i]);
            }

            auto res = envs[i]->step(static_cast<int>(a_acc[i]));
            player_state[i].step_reward(res.reward);

            if (res.done) {
                player_state[i].flush_on_terminal(i, *buffer_);
                auto rr = envs[i]->reset();
                next_obs[i]  = rr.observation;
                next_mask[i] = rr.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
                did_reset[i] = 1;
            } else {
                next_obs[i]  = res.observation;
                next_mask[i] = res.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
            }
        }

        // Re-roll opponent assignment for envs that just reset.
        for (int i = 0; i < N; ++i) {
            if (did_reset[i]) roll_episode_assignment(i);
        }

        cur_obs    = next_obs.to(device_);
        cur_mask   = next_mask.to(device_);
        cur_player = next_player.to(device_);
    }

    // Drain remaining pending transitions; these are truncations, not
    // terminals — compute_returns will bootstrap them from V(carry_obs).
    for (int i = 0; i < N; ++i) {
        player_state[i].flush_on_rollout_end(i, *buffer_);
    }

    auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();
    auto bootstrap = build_bootstrap(network_, cur_obs, cur_player_cpu, player_state);
    buffer_->compute_returns(cfg_.gamma, cfg_.gae_lambda,
                             bootstrap.values, bootstrap.terminal);

    carry_obs_            = cur_obs.to(torch::kCPU);
    carry_legal_mask_     = cur_mask.to(torch::kCPU);
    carry_current_player_ = cur_player_cpu;
    // Reset carry_done_ — per-player episode-boundary state is re-seeded at
    // the next rollout's start from this tensor, and we've already flushed
    // the pending transitions whose done flag it would have affected.
    carry_done_.zero_();

    global_step_ += cfg_.num_envs * cfg_.num_steps;
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::ensure_step_pool() {
    if (step_pool_) return;
    const int hw = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int n  = std::min(cfg_.num_envs, hw);
    step_pool_   = std::make_unique<StepThreadPool>(n);
}

// ─────────────────────────────────────────────────────────────────────────────
// Threadpool rollout — full-batch forward on the main thread, env stepping
// fanned out across a persistent worker pool with one parallel_for per step.
// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::collect_rollout_threadpool() {
    network_->eval();
    torch::NoGradGuard no_grad;

    ensure_step_pool();

    buffer_->clear();
    prepare_rollout_pool_ids();

    auto& envs = vec_env_->envs_mut();
    const int N = cfg_.num_envs;
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();

    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    // Per-env per-player rollout state. Seeded from carry_done_ so the first
    // transition after a previous-rollout reset carries the new-episode marker.
    std::vector<PlayerRolloutState> player_state(N);
    for (int i = 0; i < N; ++i) {
        const float d = carry_done_.accessor<float, 1>()[i];
        player_state[i].next_done_flag[0] = d;
        player_state[i].next_done_flag[1] = d;
    }

    auto cur_obs    = carry_obs_.to(device_);
    auto cur_mask   = carry_legal_mask_.to(device_);
    auto cur_player = carry_current_player_.to(device_);

    std::vector<uint8_t> did_reset(N, 0);

    for (int step = 0; step < cfg_.num_steps; ++step) {
        // Full-batch forward on main thread (single inference call per step).
        auto ar = network_->get_action(cur_obs, cur_mask);

        auto actions_cpu    = ar.action.to(torch::kCPU).contiguous();
        auto logp_cpu       = ar.log_prob.to(torch::kCPU).contiguous();
        auto value_cpu      = ar.value.to(torch::kCPU).contiguous();
        auto cur_obs_cpu    = cur_obs.to(torch::kCPU).contiguous();
        auto cur_mask_cpu   = cur_mask.to(torch::kCPU).contiguous();
        auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();

        // Override actions for envs where a pool snapshot is acting as the
        // non-learner seat. No-op when the pool is disabled or empty.
        apply_pool_overrides(cur_obs, cur_mask, cur_player_cpu, actions_cpu);

        auto a_acc  = actions_cpu.accessor<int64_t, 1>();
        auto lp_acc = logp_cpu.accessor<float, 1>();
        auto v_acc  = value_cpu.accessor<float, 1>();
        auto cp_acc = cur_player_cpu.accessor<int32_t, 1>();

        auto next_obs    = torch::zeros({N, D}, f_cpu);
        auto next_mask   = torch::zeros({N, A}, f_cpu);
        auto next_player = torch::zeros({N},    i32_cpu);
        auto np_acc      = next_player.accessor<int32_t, 1>();

        std::fill(did_reset.begin(), did_reset.end(), 0);

        // Parallel env stepping. Each i lands in exactly one worker, so the
        // env, player_state[i], and next_* tensor slots at [i] are touched by
        // one thread only. RolloutBuffer pushes at (player, i) are disjoint
        // across i so they're also safe — no locks needed.
        step_pool_->parallel_for(N, [&](int i) {
            const int      acting = cp_acc[i];
            const uint64_t op_id  = opp_id_[i];
            // Pool active for this env → record only the learner seat.
            // Pure self-play (op_id == 0) → record both seats as before.
            const bool record = (op_id == 0) || (acting == learner_seat_[i]);

            if (record) {
                player_state[i].record_step(
                    acting, i, *buffer_,
                    cur_obs_cpu[i], cur_mask_cpu[i],
                    a_acc[i], lp_acc[i], v_acc[i]);
            }

            auto res = envs[i]->step(static_cast<int>(a_acc[i]));
            player_state[i].step_reward(res.reward);

            if (res.done) {
                player_state[i].flush_on_terminal(i, *buffer_);
                auto rr = envs[i]->reset();
                next_obs[i]  = rr.observation;
                next_mask[i] = rr.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
                did_reset[i] = 1;
            } else {
                next_obs[i]  = res.observation;
                next_mask[i] = res.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
            }
        });

        // Re-roll opponent assignments serially: episode_rng_ isn't thread-safe.
        for (int i = 0; i < N; ++i) {
            if (did_reset[i]) roll_episode_assignment(i);
        }

        cur_obs    = next_obs.to(device_);
        cur_mask   = next_mask.to(device_);
        cur_player = next_player.to(device_);
    }

    // Drain remaining pending transitions; these are truncations, not
    // terminals — compute_returns will bootstrap them from V(carry_obs).
    for (int i = 0; i < N; ++i) {
        player_state[i].flush_on_rollout_end(i, *buffer_);
    }

    auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();
    auto bootstrap = build_bootstrap(network_, cur_obs, cur_player_cpu, player_state);
    buffer_->compute_returns(cfg_.gamma, cfg_.gae_lambda,
                             bootstrap.values, bootstrap.terminal);

    carry_obs_            = cur_obs.to(torch::kCPU);
    carry_legal_mask_     = cur_mask.to(torch::kCPU);
    carry_current_player_ = cur_player_cpu;
    carry_done_.zero_();

    global_step_ += cfg_.num_envs * cfg_.num_steps;
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic N-way interleaved benchmark. Iters of each strategy are interleaved
// per round so they all see a similar distribution of env states / cache temp.
// ─────────────────────────────────────────────────────────────────────────────
PPOTrainer::BenchmarkResult
PPOTrainer::benchmark_strategies(
    const std::vector<std::pair<std::string, RolloutFn>>& strategies,
    int iters, int warmup, bool verbose) {
    using clock = std::chrono::steady_clock;
    using ms    = std::chrono::duration<double, std::milli>;

    init_carry_state();

    const int saved_update_idx = update_idx_;
    update_idx_ = 1;  // not a multiple of 10

    auto run_one = [&](const RolloutFn& fn) {
        auto t0 = clock::now();
        fn(*this);
        return ms(clock::now() - t0).count();
    };

    // Interleaved warmup so caches/threads stabilise for every path.
    for (int i = 0; i < warmup; ++i) {
        for (const auto& s : strategies) s.second(*this);
    }

    std::vector<std::vector<double>> samples(strategies.size());
    for (auto& v : samples) v.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        for (size_t s = 0; s < strategies.size(); ++s) {
            samples[s].push_back(run_one(strategies[s].second));
        }
    }

    update_idx_ = saved_update_idx;

    auto pack = [](std::vector<double> v) -> BenchmarkResult::Stats {
        std::sort(v.begin(), v.end());
        const double sum    = std::accumulate(v.begin(), v.end(), 0.0);
        const double mean   = sum / static_cast<double>(v.size());
        const double median = v[v.size() / 2];
        const double p95    = v[std::min(v.size() - 1,
                                         static_cast<size_t>(v.size() * 0.95))];
        return {v.front(), median, mean, p95, v.back()};
    };

    BenchmarkResult r;
    r.num_envs  = cfg_.num_envs;
    r.num_steps = cfg_.num_steps;
    r.iters     = iters;
    r.samples   = cfg_.num_envs * cfg_.num_steps;
    r.by_strategy.reserve(strategies.size());
    for (size_t s = 0; s < strategies.size(); ++s) {
        r.by_strategy.emplace_back(strategies[s].first, pack(samples[s]));
    }

    if (verbose) {
        // Use the slowest median as the speedup baseline so every printed
        // multiplier is ≥ 1×.
        const auto* slowest = &r.by_strategy.front().second;
        for (const auto& kv : r.by_strategy)
            if (kv.second.median_ms > slowest->median_ms) slowest = &kv.second;

        std::cout << "\n══════════ rollout benchmark ══════════\n"
                  << "  iters=" << iters << "  warmup=" << warmup
                  << "  num_envs=" << r.num_envs
                  << "  num_steps=" << r.num_steps
                  << "  samples/rollout=" << r.samples << "\n"
                  << "  ─────────────────────────────────────\n"
                  << std::fixed << std::setprecision(2);
        for (const auto& kv : r.by_strategy) {
            const auto& name = kv.first;
            const auto& s    = kv.second;
            std::cout << "  " << std::setw(12) << std::left << name << std::right
                      << "  min=" << s.min_ms    << "ms"
                      << "  med=" << s.median_ms << "ms"
                      << "  mean=" << s.mean_ms  << "ms"
                      << "  p95="  << s.p95_ms   << "ms"
                      << "  max="  << s.max_ms   << "ms"
                      << "  spd="  << (slowest->median_ms / s.median_ms) << "x"
                      << "  us/samp=" << (s.median_ms * 1e3 / r.samples) << "\n";
        }
        std::cout.unsetf(std::ios::fixed);
        std::cout << "═══════════════════════════════════════\n\n";
    }
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience wrapper: A/B-time the three built-in strategies.
// ─────────────────────────────────────────────────────────────────────────────
PPOTrainer::BenchmarkResult
PPOTrainer::benchmark_rollouts(int iters, int warmup, bool verbose) {
    return benchmark_strategies({
        {"serial",     [](PPOTrainer& t) { t.collect_rollout_serial();     }},
        {"threadpool", [](PPOTrainer& t) { t.collect_rollout_threadpool(); }},
    }, iters, warmup, verbose);
}

// ─────────────────────────────────────────────────────────────────────────────
// scaling_benchmark — sweep num_envs, build a fresh trainer per row, summarise.
// ─────────────────────────────────────────────────────────────────────────────
void scaling_benchmark(IPokerEnvironmentFactory& factory,
                       const BetConfig& bet_cfg,
                       PPOConfig ppo_cfg,
                       torch::Device device,
                       const std::vector<int>& env_counts,
                       int iters, int warmup) {
    std::vector<PPOTrainer::BenchmarkResult> rows;
    rows.reserve(env_counts.size());

    for (int n : env_counts) {
        if (n <= 0) continue;
        std::cout << "[scaling] num_envs=" << n << " — building trainer & timing..."
                  << std::flush;
        ppo_cfg.num_envs = n;
        PPOTrainer trainer(factory, bet_cfg, ppo_cfg, device);
        rows.push_back(trainer.benchmark_rollouts(iters, warmup, /*verbose=*/false));
        std::cout << " done\n";
    }

    if (rows.empty()) return;

    // ── Summary table ───────────────────────────────────────────────────
    auto fmt = [](double v, int w, int p) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(p) << std::setw(w) << v;
        return os.str();
    };
    auto col = [](const std::string& s, int w) {
        std::ostringstream os;
        os << std::setw(w) << s;
        return os.str();
    };

    // All rows share the same strategy ordering — pull names from row 0.
    const auto& strats = rows.front().by_strategy;

    std::cout << "\n═════════════════ rollout scaling benchmark ═════════════════\n"
              << "  iters=" << iters << "  warmup=" << warmup
              << "  num_steps=" << ppo_cfg.num_steps
              << "  device=" << (device.is_cuda() ? "cuda" : "cpu") << "\n"
              << "  ───────────────────────────────────────────────────────────\n"
              << "  " << col("num_envs", 8)
              << "  " << col("samples", 8);
    for (const auto& kv : strats) std::cout << "  " << col(kv.first + "_ms",   12);
    for (const auto& kv : strats) std::cout << "  " << col(kv.first + "_us/s", 12);
    std::cout << "\n  ───────────────────────────────────────────────────────────\n";

    for (const auto& r : rows) {
        std::cout << "  " << col(std::to_string(r.num_envs), 8)
                  << "  " << col(std::to_string(r.samples),  8);
        for (const auto& kv : r.by_strategy)
            std::cout << "  " << fmt(kv.second.median_ms, 12, 2);
        for (const auto& kv : r.by_strategy)
            std::cout << "  " << fmt(kv.second.median_ms * 1e3 / r.samples, 12, 2);
        std::cout << "\n";
    }
    std::cout << "═══════════════════════════════════════════════════════════\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
PPOTrainer::UpdateStats PPOTrainer::update() {
    network_->train();

    auto batch = buffer_->flatten();
    int B = batch.obs.size(0);

    // Buffer already lives on device — no copy needed
    auto& b_obs     = batch.obs;
    auto& b_actions = batch.actions;
    auto& b_logp    = batch.log_probs;
    auto  b_adv     = batch.advantages;  // copy: normalisation mutates it
    auto& b_ret     = batch.returns;
    auto& b_val     = batch.values;
    auto& b_masks   = batch.legal_masks;

    auto stat_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    auto total_policy_loss = torch::zeros({}, stat_opts);
    auto total_value_loss  = torch::zeros({}, stat_opts);
    auto total_entropy     = torch::zeros({}, stat_opts);
    auto total_approx_kl   = torch::zeros({}, stat_opts);
    auto total_clip_frac   = torch::zeros({}, stat_opts);
    int  num_updates = 0;

    for (int epoch = 0; epoch < cfg_.update_epochs; ++epoch) {
        // Shuffle
        auto indices = torch::randperm(B, torch::TensorOptions().dtype(torch::kInt64).device(device_));

        for (int start = 0; start < B; start += cfg_.minibatch_size()) {
            int end = std::min(start + cfg_.minibatch_size(), B);
            auto mb_idx = indices.slice(0, start, end);

            auto mb_obs     = b_obs.index_select(0, mb_idx);
            auto mb_actions = b_actions.index_select(0, mb_idx);
            auto mb_logp    = b_logp.index_select(0, mb_idx);
            auto mb_adv     = b_adv.index_select(0, mb_idx);
            auto mb_ret     = b_ret.index_select(0, mb_idx);
            auto mb_val     = b_val.index_select(0, mb_idx);
            auto mb_masks   = b_masks.index_select(0, mb_idx);

            // Normalise advantages
            if (cfg_.norm_advantages && mb_adv.size(0) > 1) {
                mb_adv = (mb_adv - mb_adv.mean()) /
                         (mb_adv.std() + 1e-8f);
            }

            // Actor forward: get new log_probs and entropy for stored actions
            // Critic forward: get new value estimates
            auto er = network_->evaluate(mb_obs, mb_masks, mb_actions);

            // ── policy loss (clipped surrogate) ─────────────────────
            auto logratio = er.log_prob - mb_logp;
            auto ratio    = logratio.exp();

            auto pg_loss1 = -mb_adv * ratio;
            auto pg_loss2 = -mb_adv * torch::clamp(
                ratio, 1.0f - cfg_.clip_coef, 1.0f + cfg_.clip_coef);
            auto pg_loss  = torch::max(pg_loss1, pg_loss2).mean();

            // ── value loss ──────────────────────────────────────────
            torch::Tensor v_loss;
            if (cfg_.clip_vloss) {
                auto v_clipped = mb_val + torch::clamp(
                    er.value - mb_val,
                    -cfg_.clip_coef, cfg_.clip_coef);
                auto v_loss_unclipped = (er.value - mb_ret).pow(2);
                auto v_loss_clipped   = (v_clipped - mb_ret).pow(2);
                v_loss = 0.5f * torch::max(v_loss_unclipped,
                                           v_loss_clipped).mean();
            } else {
                v_loss = 0.5f * (er.value - mb_ret).pow(2).mean();
            }

            // ── entropy bonus ───────────────────────────────────────
            auto entropy_loss = er.entropy.mean();

            // ── total loss ──────────────────────────────────────────
            auto loss = pg_loss
                      - cfg_.ent_coef * entropy_loss
                      + cfg_.vf_coef  * v_loss;

            optimizer_->zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(network_->parameters(), cfg_.max_grad_norm);
            optimizer_->step();

            // ── stats (accumulate on device; .item() called once after loop) ──
            {
                torch::NoGradGuard ng;
                total_policy_loss += pg_loss.detach();
                total_value_loss  += v_loss.detach();
                total_entropy     += entropy_loss.detach();
                total_approx_kl   += ((ratio.detach() - 1.0f) - logratio.detach()).mean();
                total_clip_frac   += ((ratio.detach() - 1.0f).abs() > cfg_.clip_coef)
                                     .to(torch::kFloat32).mean();
            }
            ++num_updates;
        }
    }

    // ── explained variance ──────────────────────────────────────────────
    float explained_var;
    {
        torch::NoGradGuard ng;
        auto var_y = b_ret.var();
        auto var_e = (b_ret - b_val).var();
        explained_var = (var_y.item<float>() < 1e-8f)
            ? -1.0f
            : 1.0f - var_e.item<float>() / (var_y.item<float>() + 1e-8f);
    }

    // ── return stats; train() fills in timings and invokes log_cb_ ──────
    float n  = static_cast<float>(std::max(1, num_updates));
    float lr = cfg_.learning_rate;
    if (cfg_.anneal_lr) {
        float frac = 1.0f - static_cast<float>(update_idx_) / cfg_.num_updates();
        lr = cfg_.learning_rate * frac;
    }
    // Single device→host sync for all stats
    return UpdateStats{
        update_idx_,
        global_step_,
        total_policy_loss.item<float>() / n,
        total_value_loss.item<float>()  / n,
        total_entropy.item<float>()     / n,
        total_approx_kl.item<float>()   / n,
        total_clip_frac.item<float>()   / n,
        explained_var,
        lr,
        0.0,  // rollout_ms — set by train()
        0.0,  // update_ms  — set by train()
    };
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::save(const std::string& path) {
    torch::save(network_, path);
}

void PPOTrainer::load(const std::string& path) {
    torch::load(network_, path);
}

} // namespace poker_ppo
