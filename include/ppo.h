#pragma once

// PPO trainer. Owns network + optimiser, drives RolloutCollector +
// OpponentManager. Self-play: same ActorCritic plays both seats

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

    // Small wrappers over collector_->collect(strategy).
    void collect_rollout_serial();
    void collect_rollout_threadpool();

    struct BenchmarkResult {
        struct Stats { double min_ms, median_ms, mean_ms, p95_ms, max_ms; };
        int num_envs;
        int num_steps;
        int iters;
        int samples;
        // (name, stats)
        std::vector<std::pair<std::string, Stats>> by_strategy;

        const Stats* find(const std::string& name) const {
            for (const auto& [n, s] : by_strategy) if (n == name) return &s;
            return nullptr;
        }
    };

    // A/B-time the two built-in strategies
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

private:
    [[nodiscard]] UpdateStats update();

    static constexpr const PPOConfig& cfg_      = config::kPPOConfig;
    static constexpr const BetConfig& bet_cfg_  = config::kBetConfig;
    torch::Device device_;

    ActorCritic                          network_;
    std::unique_ptr<torch::optim::Adam>  optimizer_;
    std::unique_ptr<RolloutCollector>    collector_;
    std::unique_ptr<OpponentManager>     opp_mgr_;

    // MMD magnet frozen snapshot of `network_`, refreshed every
    // `cfg_.magnet_update_every` updates. Null when `cfg_.kl_coef == 0`
    ActorCritic                          magnet_{nullptr};

    int update_idx_ = 0;

    LogCallback log_cb_;
    Strategy    strategy_ = Strategy::Threadpool;
};

} // namespace poker_ppo
