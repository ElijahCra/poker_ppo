#pragma once
//
// PPO trainer. Owns network + optimiser, drives RolloutCollector +
// OpponentManager. Self-play: same ActorCritic plays both seats; rewards
// are recorded from the acting player's perspective (sign-flipped on seat 1).
//

#include "config.h"
#include "environment.h"
#include "network.h"
#include "optim.h"
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
        int     update;
        int64_t global_step;
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

    // Full training-state checkpoint (network + magnet + Adam moments +
    // update/step counters) for resuming a cut-off run. Written atomically
    // to `dir` (a sibling .tmp dir is renamed into place). The opponent
    // pool is NOT persisted — it refills from empty on resume (bounded
    // self-play blip; p_use_pool is small). save_checkpoint is also called
    // periodically from train() when set_checkpoint() armed a cadence.
    void save_checkpoint(const std::string& dir);
    // Returns true if a checkpoint was found and loaded; sets the resume
    // point so train() continues from the saved update.
    bool load_checkpoint(const std::string& dir);
    // Arm periodic checkpointing inside train(): write to `dir` every
    // `every_steps` env steps (0 disables).
    void set_checkpoint(const std::string& dir, int64_t every_steps) {
        ckpt_dir_ = dir;
        ckpt_every_steps_ = every_steps;
    }

private:
    [[nodiscard]] UpdateStats update();

    // Lazily capture one full-size minibatch's forward+loss+backward into
    // a CUDA graph (static inputs ug_*, static grads ug_grads_). The
    // optimiser stays eager: lr anneals and Adam's bias corrections change
    // every step, and scalars are baked into captured kernels. Replay
    // requires minibatch_size() rows — partial tail minibatches run eager.
    // Returns true when the graph is ready. POKER_PPO_NO_UPDATE_GRAPH=1
    // disables; capture failure falls back to eager permanently.
    bool ensure_update_graph(const torch::Tensor& b_obs,
                             const torch::Tensor& b_actions,
                             const torch::Tensor& b_logp,
                             const torch::Tensor& b_adv,
                             const torch::Tensor& b_ret,
                             const torch::Tensor& b_val,
                             const torch::Tensor& b_masks,
                             const torch::Tensor& magnet_logp_all,
                             const torch::Tensor& mb_idx,
                             bool  use_amp,
                             float ent_coef_now);

    static inline constexpr const PPOConfig& cfg_      = config::kPPOConfig;
    static inline constexpr const BetConfig& bet_cfg_  = config::kBetConfig;
    torch::Device device_;

    // Usually cfg_.num_envs; overridable at construction via the
    // POKER_PPO_NUM_ENVS env var for throughput sweeps (benchmark only —
    // update() minibatching still assumes cfg_.num_envs, so don't train
    // with an override, change the constexpr in config.h instead).
    int num_envs_;

    ActorCritic                          network_;
    std::unique_ptr<ForeachAdam>         optimizer_;
    std::unique_ptr<RolloutCollector>    collector_;
    std::unique_ptr<OpponentManager>     opp_mgr_;

    // MMD magnet — frozen snapshot of `network_`, refreshed every
    // `cfg_.magnet_update_every` updates. Null when `cfg_.kl_coef == 0`
    // (vanilla self-play PPO; the regulariser is bypassed entirely).
    ActorCritic                          magnet_{nullptr};

    int     update_idx_              = 0;
    int     start_update_            = 0;   // resume point (0 = fresh)
    int64_t last_magnet_refresh_step_ = 0;

    // Periodic-checkpoint state (see set_checkpoint / save_checkpoint).
    std::string ckpt_dir_;
    int64_t     ckpt_every_steps_ = 0;
    int64_t     last_ckpt_step_   = 0;

    // CUDA-graphed minibatch step (see ensure_update_graph).
    enum class UGraphState { Unset, Ready, Failed };
    UGraphState                           ugraph_state_ = UGraphState::Unset;
    std::unique_ptr<at::cuda::CUDAGraph>  ugraph_;
    // Static inputs, refilled per minibatch via index_select_out.
    torch::Tensor ug_obs_, ug_actions_, ug_logp_, ug_adv_, ug_ret_, ug_val_,
                  ug_masks_, ug_mlogp_;
    // Static outputs: detached stat scalars + gradients (graph pool).
    torch::Tensor ug_pg_, ug_vl_, ug_ent_, ug_kl_, ug_clip_;
    std::vector<torch::Tensor> ug_grads_;

    LogCallback log_cb_;
    Strategy    strategy_ = Strategy::Threadpool;
};

} // namespace poker_ppo
