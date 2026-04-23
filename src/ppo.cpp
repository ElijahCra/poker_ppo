#include "ppo.h"
#include "rollout_coro.h"
#include "rollout_pool.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>

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

VectorizedEnv::BatchStepResult
VectorizedEnv::step(const std::vector<int>& actions) {
    int N = num_envs();
    rewards_buf_.zero_();
    dones_buf_.zero_();

    for (int i = 0; i < N; ++i) {
        int acting_player = envs_[i]->current_player();
        auto result = envs_[i]->step(actions[i]);

        float r = result.reward;
        if (acting_player == 1) r = -r;

        if (result.done) {
            rewards_buf_[i] = r;
            dones_buf_[i]   = 1.0f;
            auto rr = envs_[i]->reset();
            obs_buf_[i]     = rr.observation;
            masks_buf_[i]   = rr.legal_action_mask;
            players_buf_[i] = envs_[i]->current_player();
        } else {
            obs_buf_[i]     = result.observation;
            rewards_buf_[i] = r;
            masks_buf_[i]   = result.legal_action_mask;
            players_buf_[i] = envs_[i]->current_player();
        }
    }

    return {obs_buf_.clone(), rewards_buf_.clone(),
            dones_buf_.clone(), masks_buf_.clone(), players_buf_.clone()};
}

// ═════════════════════════════════════════════════════════════════════════════
// PPOTrainer
// ═════════════════════════════════════════════════════════════════════════════

PPOTrainer::PPOTrainer(IPokerEnvironmentFactory& env_factory,
                       const BetConfig& bet_cfg,
                       const PPOConfig& ppo_cfg,
                       torch::Device device)
    : cfg_(ppo_cfg), bet_cfg_(bet_cfg), device_(device),
      network_(nullptr)
{
    // Create vectorized environment
    vec_env_ = std::make_unique<VectorizedEnv>(
        env_factory, bet_cfg, cfg_.num_envs);

    int obs_dim      = vec_env_->obs_dim();
    int action_count = vec_env_->action_count();

    // Create network
    network_ = ActorCritic(obs_dim, action_count, cfg_.hidden_dim, cfg_.num_layers);
    network_->to(device_);

    // Optimiser
    optimizer_ = std::make_unique<torch::optim::Adam>(
        network_->parameters(),
        torch::optim::AdamOptions(cfg_.learning_rate));

    // Rollout buffer (lives on device — avoids CPU round-trip each rollout)
    buffer_ = std::make_unique<RolloutBuffer>(
        cfg_.num_steps, cfg_.num_envs, obs_dim, action_count, device_);
}

PPOTrainer::~PPOTrainer() = default;

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
        // Anneal learning rate
        if (cfg_.anneal_lr) {
            float frac = 1.0f - static_cast<float>(update_idx_) / total_updates;
            float lr   = cfg_.learning_rate * frac;
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
                      << "  update=" << stats.update_ms << "ms\n";
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::collect_rollout_coroutine() {
    network_->eval();
    torch::NoGradGuard no_grad;

    using clock = std::chrono::steady_clock;
    using ms    = std::chrono::duration<double, std::milli>;

    // Reach back into VectorizedEnv for the raw env vector. We need per-env
    // access so the coroutines can step individually.
    std::vector<std::unique_ptr<IPokerEnvironment>> dummy_envs_unused;  // for signature
    // (RolloutScheduler keeps a reference, so we need a stable vector. We'll
    // vend one from VectorizedEnv via a small helper.)

    // Acquire the owning vector of envs from VectorizedEnv.
    auto& envs = vec_env_->envs_mut();

    // Batch-inference tuning. min_batch ≤ num_envs so the processor fires
    // even when not every coroutine has submitted yet; max_wait caps latency
    // for the tail end of a step.
    // Half-wave pipelining: firing on N/2 lets fast-stepping envs resubmit
    // their next request while the processor is still running the forward
    // on the slower half, overlapping env work with inference. Firing on N
    // creates a full-wave barrier with zero overlap — worse despite fewer
    // dispatches. If forward_ns > worker_busy_ns (we're forward-bound),
    // consider dropping further (N/4) to pipeline more aggressively.
    const std::size_t min_batch =
        std::max<std::size_t>(1, static_cast<std::size_t>(cfg_.num_envs) );
    const auto max_wait = std::chrono::microseconds(500);

    // Worker pool: each owns a shard of envs and resumes their coroutines
    // in parallel. Overprovisioning past ~num_envs buys nothing (at most
    // one coro per env can be ready at a time on a given shard). Defaults
    // to 4 — tune based on core count and env-step cost.
    const int num_workers = std::min(
        cfg_.num_envs, 4);
        //std::max(1, static_cast<int>(std::thread::hardware_concurrency()) / 2));

    auto t0 = clock::now();

    RolloutScheduler scheduler(
        network_, device_,
        envs,
        *buffer_,
        cfg_.num_steps,
        vec_env_->obs_dim(),
        vec_env_->action_count(),
        carry_obs_, carry_legal_mask_, carry_current_player_, carry_done_,
        min_batch, max_wait,
        num_workers,
        cfg_.reward_scale);

    scheduler.run();

    auto t1 = clock::now();

    scheduler.flush_to_buffer();

    // GAE bootstrap tensors live on CPU in the scheduler; move once to device.
    auto next_value  = scheduler.bootstrap_value().to(device_);
    auto next_done   = scheduler.bootstrap_done().to(device_);
    auto next_player = scheduler.bootstrap_player().to(device_);

    buffer_->compute_returns(next_value, next_done, next_player,
                             cfg_.gamma, cfg_.gae_lambda);

    auto t2 = clock::now();

    global_step_ += cfg_.num_envs * cfg_.num_steps;

    if (update_idx_ % 10 == 0) {
        const double fwd_ms  = scheduler.forward_ns() / 1e6;
        const int    W       = scheduler.num_workers();
        const double wait_ms = (W > 0)
            ? (scheduler.worker_wait_ns() / 1e6) / static_cast<double>(W)
            : 0.0;
        std::cout << "  rollout breakdown:"
                  << " coro_run=" << ms(t1 - t0).count() << "ms"
                  << " flush+gae=" << ms(t2 - t1).count() << "ms"
                  << "  forward=" << fwd_ms << "ms"
                  << "  worker_wait_avg=" << wait_ms << "ms"
                  << " (W=" << W << ")\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Serial rollout — single-thread loop matching commit b601e0e.
// Adapted to current state shape: carry_* tensors live on CPU, RolloutBuffer
// lives on device. We move per-step batches across the device boundary the
// same way the old code did (cur_* held on device, scratch allocated CPU,
// rewards/dones built CPU and pushed to device for buffer insert).
// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::collect_rollout_serial() {
    network_->eval();
    torch::NoGradGuard no_grad;

    auto& envs = vec_env_->envs_mut();
    const int N = cfg_.num_envs;
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();

    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    // Pull carry state onto device for the forward pass.
    auto cur_obs    = carry_obs_.to(device_);
    auto cur_mask   = carry_legal_mask_.to(device_);
    auto cur_done   = carry_done_.to(device_);
    auto cur_player = carry_current_player_.to(device_);

    for (int step = 0; step < cfg_.num_steps; ++step) {
        // Full-batch forward (the whole point of the old design).
        auto ar = network_->get_action(cur_obs, cur_mask);

        auto actions_cpu = ar.action.to(torch::kCPU).contiguous();
        auto a_acc       = actions_cpu.accessor<int64_t, 1>();
        auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();
        auto cp_acc      = cur_player_cpu.accessor<int32_t, 1>();

        // Per-step CPU scratch for next state.
        auto next_obs    = torch::zeros({N, D}, f_cpu);
        auto next_mask   = torch::zeros({N, A}, f_cpu);
        auto next_done   = torch::zeros({N},    f_cpu);
        auto next_player = torch::zeros({N},    i32_cpu);
        auto rewards     = torch::zeros({N},    f_cpu);

        auto rew_acc    = rewards.accessor<float, 1>();
        auto nd_acc     = next_done.accessor<float, 1>();
        auto np_acc     = next_player.accessor<int32_t, 1>();

        for (int i = 0; i < N; ++i) {
            const int acting = cp_acc[i];
            auto res = envs[i]->step(static_cast<int>(a_acc[i]));

            float r = res.reward;
            if (acting == 1) r = -r;
            rew_acc[i] = r * cfg_.reward_scale;

            if (res.done) {
                nd_acc[i] = 1.0f;
                auto rr = envs[i]->reset();
                next_obs[i]  = rr.observation;
                next_mask[i] = rr.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
            } else {
                nd_acc[i] = 0.0f;
                next_obs[i]  = res.observation;
                next_mask[i] = res.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
            }
        }

        // Store PRE-step transition. Buffer lives on device — push there.
        buffer_->insert(step,
                        cur_obs,
                        ar.action,
                        ar.log_prob,
                        rewards.to(device_),
                        cur_done,
                        ar.value,
                        cur_mask,
                        cur_player.to(device_));

        // Advance to next state on device.
        cur_obs    = next_obs.to(device_);
        cur_mask   = next_mask.to(device_);
        cur_done   = next_done.to(device_);
        cur_player = next_player.to(device_);
    }

    // Bootstrap V(s_T).
    auto [_, next_value] = network_->forward(cur_obs);
    buffer_->compute_returns(next_value, cur_done, cur_player,
                             cfg_.gamma, cfg_.gae_lambda);

    // Persist back to carry state so subsequent rollouts continue from here.
    carry_obs_            = cur_obs.to(torch::kCPU);
    carry_legal_mask_     = cur_mask.to(torch::kCPU);
    carry_done_           = cur_done.to(torch::kCPU);
    carry_current_player_ = cur_player.to(torch::kCPU);

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
// Cheaper synchronisation than the coroutine path (no per-step scheduler
// teardown, no shared_ptr per request) but no inference/env overlap; well
// suited when forward dominates and env work is tiny.
// ─────────────────────────────────────────────────────────────────────────────
void PPOTrainer::collect_rollout_threadpool() {
    network_->eval();
    torch::NoGradGuard no_grad;

    ensure_step_pool();

    auto& envs = vec_env_->envs_mut();
    const int N = cfg_.num_envs;
    const int D = vec_env_->obs_dim();
    const int A = vec_env_->action_count();

    auto f_cpu   = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto i32_cpu = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

    auto cur_obs    = carry_obs_.to(device_);
    auto cur_mask   = carry_legal_mask_.to(device_);
    auto cur_done   = carry_done_.to(device_);
    auto cur_player = carry_current_player_.to(device_);

    for (int step = 0; step < cfg_.num_steps; ++step) {
        // Full-batch forward on main thread (single inference call per step).
        auto ar = network_->get_action(cur_obs, cur_mask);

        auto actions_cpu    = ar.action.to(torch::kCPU).contiguous();
        auto cur_player_cpu = cur_player.to(torch::kCPU).contiguous();
        auto a_acc          = actions_cpu.accessor<int64_t, 1>();
        auto cp_acc         = cur_player_cpu.accessor<int32_t, 1>();

        auto next_obs    = torch::zeros({N, D}, f_cpu);
        auto next_mask   = torch::zeros({N, A}, f_cpu);
        auto next_done   = torch::zeros({N},    f_cpu);
        auto next_player = torch::zeros({N},    i32_cpu);
        auto rewards     = torch::zeros({N},    f_cpu);

        auto rew_acc = rewards.accessor<float, 1>();
        auto nd_acc  = next_done.accessor<float, 1>();
        auto np_acc  = next_player.accessor<int32_t, 1>();

        // Parallel env stepping. Each i lands in exactly one worker, so the
        // env, scratch tensors at slot [i], and accessors at [i] are touched
        // by one thread only — no env-level locks needed.
        const float rscale = cfg_.reward_scale;
        step_pool_->parallel_for(N, [&](int i) {
            const int acting = cp_acc[i];
            auto res = envs[i]->step(static_cast<int>(a_acc[i]));

            float r = res.reward;
            if (acting == 1) r = -r;
            rew_acc[i] = r * rscale;

            if (res.done) {
                nd_acc[i] = 1.0f;
                auto rr = envs[i]->reset();
                next_obs[i]  = rr.observation;
                next_mask[i] = rr.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
            } else {
                nd_acc[i] = 0.0f;
                next_obs[i]  = res.observation;
                next_mask[i] = res.legal_action_mask;
                np_acc[i]    = static_cast<int32_t>(envs[i]->current_player());
            }
        });

        buffer_->insert(step,
                        cur_obs,
                        ar.action,
                        ar.log_prob,
                        rewards.to(device_),
                        cur_done,
                        ar.value,
                        cur_mask,
                        cur_player.to(device_));

        cur_obs    = next_obs.to(device_);
        cur_mask   = next_mask.to(device_);
        cur_done   = next_done.to(device_);
        cur_player = next_player.to(device_);
    }

    auto [_, next_value] = network_->forward(cur_obs);
    buffer_->compute_returns(next_value, cur_done, cur_player,
                             cfg_.gamma, cfg_.gae_lambda);

    carry_obs_            = cur_obs.to(torch::kCPU);
    carry_legal_mask_     = cur_mask.to(torch::kCPU);
    carry_done_           = cur_done.to(torch::kCPU);
    carry_current_player_ = cur_player.to(torch::kCPU);

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

    // Suppress per-rollout breakdown logging from collect_rollout_coroutine.
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
        {"coroutine",  [](PPOTrainer& t) { t.collect_rollout_coroutine();  }},
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
