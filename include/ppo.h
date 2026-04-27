#pragma once

#include "types.h"
#include "environment.h"
#include "network.h"
#include "rollout_buffer.h"

#include <torch/torch.h>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace poker_ppo {

class StepThreadPool;  // fwd-decl; full type lives in rollout_pool.h
class OpponentPool;    // fwd-decl; full type lives in opponent_pool.h

// ─────────────────────────────────────────────────────────────────────────────
// PPOTrainer
// ─────────────────────────────────────────────────────────────────────────────
//
// Usage:
//   1. Implement IPokerEnvironment & IPokerEnvironmentFactory.
//   2. Create a PPOTrainer with your factory, BetConfig, and PPOConfig.
//   3. Call train().  Optionally register a callback for logging.
//
// The trainer runs self-play: the same ActorCritic network plays both seats.
// Rewards stored in the buffer are always from the perspective of the acting
// player (flipped sign when seat == 1).

class PPOTrainer {
public:
    /// Per-update statistics passed to the logging callback.
    struct UpdateStats {
        int    update;
        int    global_step;
        float  policy_loss;
        float  value_loss;
        float  entropy;
        float  approx_kl;
        float  clip_fraction;
        float  explained_variance;
        float  learning_rate;
        double rollout_ms;   // wall time for the rollout-collection phase
        double update_ms;    // wall time for the PPO update phase
    };

    using LogCallback = std::function<void(const UpdateStats&)>;

    PPOTrainer(IPokerEnvironmentFactory& env_factory,
               const BetConfig& bet_cfg,
               const PPOConfig& ppo_cfg,
               torch::Device device = torch::kCPU);

    // Out-of-line so the implicit destructor doesn't need StepThreadPool's
    // full definition at every PPOTrainer use site.
    ~PPOTrainer();

    /// Pluggable rollout-collection strategy. Pass any callable matching
    /// this signature into set_rollout_fn() to swap how train() collects.
    using RolloutFn = std::function<void(PPOTrainer&)>;

    /// Run the full training loop.
    /// Uses the rollout function set via set_rollout_fn(); defaults to the
    void train();

    /// Choose the rollout strategy used by train().
    /// Convenience helpers for the three built-in strategies are below.
    void set_rollout_fn(RolloutFn fn) { rollout_fn_ = std::move(fn); }

    // ── Built-in rollout strategies ─────────────────────────────────────
    // Each is a self-contained collect_* method that fills the buffer for
    // one rollout. They are public so they can be wired into set_rollout_fn,
    // benchmarked, or invoked manually.
    void collect_rollout_serial();      // single-thread loop, matches b601e0e
    void collect_rollout_threadpool();  // persistent thread pool, batched fwd

    /// Aggregate rollout-time stats for one strategy.
    struct BenchmarkResult {
        struct Stats { double min_ms, median_ms, mean_ms, p95_ms, max_ms; };
        int num_envs;
        int num_steps;
        int iters;
        int samples;        // num_envs × num_steps
        // Ordered list of (strategy_name, stats). Order matches input order
        // for benchmark_strategies(); for benchmark_rollouts() the order is
        // serial -> threadpool.
        std::vector<std::pair<std::string, Stats>> by_strategy;

        const Stats* find(const std::string& name) const {
            for (const auto& [n, s] : by_strategy) if (n == name) return &s;
            return nullptr;
        }
    };

    /// A/B-time the three built-in strategies (serial, threadpool).
    /// Iterations are interleaved per round so each strategy faces a similar
    /// env distribution. Does not call update() — measures rollout only.
    BenchmarkResult benchmark_rollouts(int iters = 20, int warmup = 3,
                                       bool verbose = true);

    /// Generalised version: bench any user-supplied list of strategies.
    BenchmarkResult benchmark_strategies(
        const std::vector<std::pair<std::string, RolloutFn>>& strategies,
        int iters = 20, int warmup = 3, bool verbose = true);

    /// Register a callback invoked after each PPO update.
    void set_log_callback(LogCallback cb) { log_cb_ = std::move(cb); }

    /// Access the trained network (e.g. for evaluation / saving).
    ActorCritic& network() { return network_; }

    /// Number of snapshots currently in the opponent pool. 0 if disabled or
    /// not yet warmed up.
    int opponent_pool_size() const;

    /// Save / load model weights.
    void save(const std::string& path);
    void load(const std::string& path);

private:
    void init_carry_state();         // resets envs and seeds carry_* tensors
    UpdateStats update();            // PPO update; returns per-update stats
    void ensure_step_pool();         // lazy-init for the threadpool strategy

    // Opponent-pool helpers. No-ops when cfg_.opp_pool.enabled == false.
    void maybe_snapshot();           // called by train() each update
    void prepare_rollout_pool_ids(); // called at the start of each rollout
    void roll_episode_assignment(int env_idx);  // re-sample learner_seat / opp_id
    void apply_pool_overrides(
        const torch::Tensor& cur_obs,        // [N, D] device
        const torch::Tensor& cur_mask,       // [N, A] device
        const torch::Tensor& cur_player_cpu, // [N] int32 CPU
        torch::Tensor& actions_cpu);         // [N] int64 CPU — modified in place

    PPOConfig    cfg_;
    BetConfig    bet_cfg_;
    torch::Device device_;

    ActorCritic  network_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::unique_ptr<VectorizedEnv>      vec_env_;
    std::unique_ptr<RolloutBuffer>      buffer_;

    // State carried between rollouts (CPU — consumed by the
    // scheduler, which does one batched transfer to device per inference).
    torch::Tensor carry_obs_;           // [num_envs, obs_dim]      float
    torch::Tensor carry_legal_mask_;    // [num_envs, action_count] float
    torch::Tensor carry_current_player_;  // [num_envs]             int32
    torch::Tensor carry_done_;          // [num_envs]               float

    int global_step_ = 0;
    int update_idx_  = 0;

    LogCallback log_cb_;

    // Pluggable rollout strategy used by train(). Defaulted in train() if
    // the caller hasn't installed one explicitly.
    RolloutFn rollout_fn_;

    // Persistent worker pool used by collect_rollout_threadpool(). Created
    // on first call; reused across rollouts to avoid per-step thread spawn.
    std::unique_ptr<StepThreadPool> step_pool_;

    // ── Opponent pool (self-play stabilisation) ─────────────────────────
    // opp_pool_ is null when cfg_.opp_pool.enabled == false. learner_seat_
    // and opp_id_ are sized num_envs and re-rolled at episode boundaries:
    //   learner_seat_[i] ∈ {0, 1}    — which seat is the live policy
    //   opp_id_[i]                   — 0: pure self-play (no override)
    //                                  else: SnapshotId of the pool member
    //                                  that plays the non-learner seat
    // Stale IDs (snapshot dropped between sample and use) fall through to
    // live policy via OpponentPool::has_id().
    std::unique_ptr<OpponentPool> opp_pool_;
    std::vector<int>              learner_seat_;
    std::vector<uint64_t>         opp_id_;
    // Pool IDs eligible for assignment in the *current* rollout. Sampled at
    // rollout start, sized at most cfg_.opp_pool.max_unique_per_rollout. Empty
    // means "fall back to full-pool sampling" (warmup, disabled, etc.).
    std::vector<uint64_t>         rollout_pool_ids_;
    std::mt19937                  episode_rng_;
};

// ─────────────────────────────────────────────────────────────────────────────
// scaling_benchmark
// ─────────────────────────────────────────────────────────────────────────────
//
// Build a fresh PPOTrainer for each value in `env_counts` and run the 3-way
// benchmark_rollouts() against it. Prints one summary table at the end so
// you can see how each strategy scales with num_envs.
//
// `ppo_cfg` is taken by value because we mutate `num_envs` per row.
// `num_steps` and the network shape stay constant across rows.

void scaling_benchmark(IPokerEnvironmentFactory& factory,
                       const BetConfig& bet_cfg,
                       PPOConfig ppo_cfg,
                       torch::Device device,
                       const std::vector<int>& env_counts,
                       int iters  = 10,
                       int warmup = 2);

} // namespace poker_ppo
