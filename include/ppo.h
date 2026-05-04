#pragma once
//
// ppo.h — PPO trainer orchestrator. Composes a RolloutCollector and an
// OpponentManager, owns the network + optimiser, runs the training loop.
// Self-play: the same ActorCritic plays both seats; buffered rewards are
// always from the acting player's perspective (sign-flipped on seat 1).
//

#include "config.h"
#include "environment.h"
#include "network.h"
#include "rollout_collector.h"

#include <torch/torch.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace poker_ppo {

class OpponentManager;    // fwd-decl; full type lives in opponent_manager.h

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

    /// Hyperparameters come from `config::kPPOConfig` and `config::kBetConfig`
    /// (compile-time constants). The constructor only takes runtime-only
    /// inputs: the env factory and the device.
    explicit PPOTrainer(IPokerEnvironmentFactory& env_factory,
                        torch::Device device = torch::kCPU);

    // Out-of-line so callers don't need OpponentManager / RolloutCollector
    // fully defined at every PPOTrainer use site.
    ~PPOTrainer();

    /// Pluggable rollout-collection strategy used by `train()`. The default
    /// is `Strategy::Threadpool`; call `set_rollout_strategy()` to change.
    using Strategy = RolloutCollector::Strategy;

    /// Run the full training loop until `total_timesteps` is reached.
    void train();

    /// Choose the rollout strategy used by `train()` and the convenience
    /// `collect_rollout_*` shims below.
    void set_rollout_strategy(Strategy s) noexcept { strategy_ = s; }

    // ── Built-in rollout entry points ───────────────────────────────────
    // Thin shims around `collector_->collect(strategy, ...)` kept so callers
    // (benchmarks, tests) don't need to know about `OpponentManager` etc.
    void collect_rollout_serial();
    void collect_rollout_threadpool();

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

    /// A/B-time the two built-in strategies (serial, threadpool).
    /// Iterations are interleaved per round so each strategy faces a
    /// similar env distribution. Does not call update().
    BenchmarkResult benchmark_rollouts(int iters = 20, int warmup = 3,
                                       bool verbose = true);

    /// Generalised version: bench any user-supplied list of strategies.
    using RolloutFn = std::function<void(PPOTrainer&)>;
    BenchmarkResult benchmark_strategies(
        const std::vector<std::pair<std::string, RolloutFn>>& strategies,
        int iters = 20, int warmup = 3, bool verbose = true);

    /// Register a callback invoked after each PPO update.
    void set_log_callback(LogCallback cb) { log_cb_ = std::move(cb); }

    /// Access the trained network (e.g. for evaluation / saving).
    ActorCritic& network() { return network_; }

    /// Number of snapshots currently in the opponent pool. 0 if disabled.
    int opponent_pool_size() const;

    /// Save / load model weights.
    void save(const std::string& path);
    void load(const std::string& path);

private:
    [[nodiscard]] UpdateStats update();   // PPO optimiser step; per-update stats

    // Compile-time hyperparameter sources. Kept as `cfg_` / `bet_cfg_`
    // references so `if constexpr (cfg_.flag)` reads cleanly; the `inline`
    // makes them non-ODR definitions.
    static inline constexpr const PPOConfig& cfg_      = config::kPPOConfig;
    static inline constexpr const BetConfig& bet_cfg_  = config::kBetConfig;
    torch::Device device_;

    ActorCritic                          network_;
    std::unique_ptr<torch::optim::Adam>  optimizer_;
    std::unique_ptr<RolloutCollector>    collector_;
    std::unique_ptr<OpponentManager>     opp_mgr_;

    int update_idx_ = 0;

    LogCallback log_cb_;
    Strategy    strategy_ = Strategy::Threadpool;
};

} // namespace poker_ppo
