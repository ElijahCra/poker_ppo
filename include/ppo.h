#pragma once

#include "types.h"
#include "environment.h"
#include "network.h"
#include "rollout_buffer.h"

#include <torch/torch.h>
#include <functional>
#include <memory>

namespace poker_ppo {

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
    };

    using LogCallback = std::function<void(const UpdateStats&)>;

    PPOTrainer(IPokerEnvironmentFactory& env_factory,
               const BetConfig& bet_cfg,
               const PPOConfig& ppo_cfg,
               torch::Device device = torch::kCPU);

    /// Run the full training loop.
    void train();

    /// Register a callback invoked after each PPO update.
    void set_log_callback(LogCallback cb) { log_cb_ = std::move(cb); }

    /// Access the trained network (e.g. for evaluation / saving).
    ActorCritic& network() { return network_; }

    /// Save / load model weights.
    void save(const std::string& path);
    void load(const std::string& path);

private:
    void collect_rollout();
    void update();

    PPOConfig    cfg_;
    BetConfig    bet_cfg_;
    torch::Device device_;

    ActorCritic  network_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::unique_ptr<VectorizedEnv>      vec_env_;
    std::unique_ptr<RolloutBuffer>      buffer_;

    // State carried between rollouts (CPU — consumed by the coroutine
    // scheduler, which does one batched transfer to device per inference).
    torch::Tensor carry_obs_;           // [num_envs, obs_dim]      float
    torch::Tensor carry_legal_mask_;    // [num_envs, action_count] float
    torch::Tensor carry_current_player_;  // [num_envs]             int32
    torch::Tensor carry_done_;          // [num_envs]               float

    int global_step_ = 0;
    int update_idx_  = 0;

    LogCallback log_cb_;
};

} // namespace poker_ppo
