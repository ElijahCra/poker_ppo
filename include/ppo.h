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
// The trainer runs self-play: a single Actor and a single Critic play both
// seats.  The Actor and Critic have fully independent parameters (no shared
// trunk), eliminating gradient interference between the policy and value
// objectives.
//
// Rewards stored in the buffer are always from the perspective of the acting
// player (flipped sign when seat == 1).  GAE correctly handles the sign
// inversion at player-switch boundaries (zero-sum).

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

    /// Access the trained networks (e.g. for evaluation / saving).
    Actor&  actor()  { return actor_; }
    Critic& critic() { return critic_; }

    /// Save / load model weights.
    void save(const std::string& path_prefix);
    void load(const std::string& path_prefix);

private:
    void collect_rollout();
    void update();

    PPOConfig    cfg_;
    BetConfig    bet_cfg_;
    torch::Device device_;

    Actor   actor_;
    Critic  critic_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
    std::unique_ptr<VectorizedEnv>      vec_env_;
    std::unique_ptr<RolloutBuffer>      buffer_;

    // State carried between rollouts
    torch::Tensor next_obs_;             // [num_envs, obs_dim]
    torch::Tensor next_done_;            // [num_envs]
    torch::Tensor next_legal_mask_;      // [num_envs, action_count]
    torch::Tensor next_current_player_;  // [num_envs]  int32

    int global_step_ = 0;
    int update_idx_  = 0;

    LogCallback log_cb_;
};

} // namespace poker_ppo
