#pragma once
//
// PPO trainer. Owns network + optimiser, drives RolloutCollector +
// OpponentManager. Self-play: same ActorCritic plays both seats; rewards
// are recorded from the acting player's perspective (sign-flipped on seat 1).
//

#include "config.h"
#include "environment.h"
#include "network.h"
#include "rollout.h"

#include <torch/torch.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace poker_ppo {

class OpponentManager;

class PPOTrainer {
public:
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
        float  cfv_loss;            // 0 when CFV aux is disabled
        double rollout_ms;
        double update_ms;
    };

    using LogCallback = std::function<void(const UpdateStats&)>;

    // Hyperparameters come from config::kPPOConfig + config::kBetConfig.
    explicit PPOTrainer(IPokerEnvironmentFactory& env_factory,
                        torch::Device device = torch::kCPU);

    ~PPOTrainer();

    using Strategy = RolloutCollector::Strategy;

    void train();

    void set_rollout_strategy(Strategy s) noexcept { strategy_ = s; }

    // Thin shims around collector_->collect(strategy, ...). Used by
    // benchmarks/tests that don't want to know about OpponentManager.
    void collect_rollout_serial();
    void collect_rollout_threadpool();

    struct BenchmarkResult {
        struct Stats { double min_ms, median_ms, mean_ms, p95_ms, max_ms; };
        int num_envs;
        int num_steps;
        int iters;
        int samples;
        // (name, stats), preserves insertion order. benchmark_rollouts()
        // emits serial then threadpool.
        std::vector<std::pair<std::string, Stats>> by_strategy;

        const Stats* find(const std::string& name) const {
            for (const auto& [n, s] : by_strategy) if (n == name) return &s;
            return nullptr;
        }
    };

    // A/B-time the two built-in strategies. Iterations interleave per round
    // so each strategy faces a similar env distribution. No update().
    BenchmarkResult benchmark_rollouts(int iters = 20, int warmup = 3,
                                       bool verbose = true);

    using RolloutFn = std::function<void(PPOTrainer&)>;
    BenchmarkResult benchmark_strategies(
        const std::vector<std::pair<std::string, RolloutFn>>& strategies,
        int iters = 20, int warmup = 3, bool verbose = true);

    void set_log_callback(LogCallback cb) { log_cb_ = std::move(cb); }

    ActorCritic& network() { return network_; }

    int opponent_pool_size() const;

    void save(const std::string& path);
    void load(const std::string& path);

    // Used for checkpoint dir resolution. If empty (default), no
    // checkpoint save/resume happens regardless of the config flag.
    void set_run_dir(std::string dir) { run_dir_ = std::move(dir); }

private:
    [[nodiscard]] UpdateStats update();

    // Synchronous loop body (original behavior).
    void train_sync_loop(int total_updates);

    // Double-buffered async loop: worker thread runs rollout on a snapshot
    // while main thread does the update; sync at iteration boundary.
    void train_async_loop(int total_updates);

    // Clone live network into snapshot_. Called at iter start before
    // dispatching the rollout to the worker.
    void refresh_snapshot();

    // Magnet/opp-pool refresh logic shared by both loops.
    void maybe_refresh_magnet();

    static inline constexpr const PPOConfig& cfg_      = config::kPPOConfig;
    static inline constexpr const BetConfig& bet_cfg_  = config::kBetConfig;
    torch::Device device_;

    ActorCritic                          network_;
    std::unique_ptr<torch::optim::Adam>  optimizer_;
    std::unique_ptr<RolloutCollector>    collector_;
    std::unique_ptr<RolloutCollector>    collector_b_;   // async: second buffer + envs
    std::unique_ptr<OpponentManager>     opp_mgr_;

    // MMD magnet — frozen snapshot of `network_`, refreshed every
    // `cfg_.magnet_update_every` updates. Null when `cfg_.kl_coef == 0`
    // (vanilla self-play PPO; the regulariser is bypassed entirely).
    ActorCritic                          magnet_{nullptr};

    // Frozen snapshot of network_ used by the async worker for rollouts.
    // Refreshed at the top of each iter so the rollout sees the most
    // recent policy minus 1-step lag. Null when async is disabled.
    ActorCritic                          snapshot_{nullptr};

    // Which collector's buffer holds the data to be consumed by the next
    // update call. The other collector is the one the worker fills.
    // Toggled at each async iter boundary.
    bool                                 current_is_a_ = true;

    // Mirror of RolloutCollector::global_step at the end of the last
    // iter — surfaced here so checkpoint save can persist it without
    // having to pick the right collector.
    int update_idx_  = 0;
    int global_step_ = 0;

    // For checkpoint save/resume. Empty disables both.
    std::string run_dir_;

    LogCallback log_cb_;
    Strategy    strategy_ = Strategy::Threadpool;
};

} // namespace poker_ppo
