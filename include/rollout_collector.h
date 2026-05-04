#pragma once
//
// rollout_collector.h — owns the rollout phase of PPO training.
//
// Responsibilities:
//   - VectorizedEnv (one env per slot, factory-built)
//   - RolloutBuffer (lives on device)
//   - StepThreadPool (lazy; only created if a Threadpool rollout is run)
//   - Per-rollout carry state (last obs/mask/player/done) between calls
//   - The rollout loop itself (serial OR threadpool, single shared body)
//   - Bootstrap + GAE return computation at rollout end
//

#include "config.h"
#include "environment.h"
#include "network.h"
#include "rollout_buffer.h"

#include <torch/torch.h>

#include <memory>

namespace poker_ppo {

class StepThreadPool;     // fwd-decl; full type in rollout_pool.h
class OpponentManager;    // fwd-decl; full type in opponent_manager.h

class RolloutCollector {
public:
    enum class Strategy { Serial, Threadpool };

    RolloutCollector(IPokerEnvironmentFactory& factory,
                     const BetConfig&          bet_cfg,
                     int                       num_envs,
                     int                       num_steps,
                     torch::Device             device);

    // Out-of-line so callers don't need StepThreadPool's full definition.
    ~RolloutCollector();

    // Reset envs + carry tensors. Call once before the training loop.
    void init_carry();

    // One rollout: fills the buffer, bootstraps, computes returns, updates
    // carry state, advances global_step_. `update_idx` is forwarded to
    // `opp_mgr` so its warmup/snapshot-cadence logic works.
    void collect(Strategy           strategy,
                 ActorCritic&       network,
                 OpponentManager&   opp_mgr,
                 int                update_idx,
                 float              gamma,
                 float              gae_lambda);

    [[nodiscard]] int                   obs_dim()      const noexcept { return vec_env_->obs_dim(); }
    [[nodiscard]] int                   action_count() const noexcept { return vec_env_->action_count(); }
    [[nodiscard]] int                   global_step()  const noexcept { return global_step_; }
    [[nodiscard]] RolloutBuffer&        buffer()       noexcept       { return *buffer_; }
    [[nodiscard]] const RolloutBuffer&  buffer()       const noexcept { return *buffer_; }

private:
    void ensure_step_pool();

    torch::Device                   device_;
    int                             num_envs_;
    int                             num_steps_;
    int                             global_step_ = 0;

    std::unique_ptr<VectorizedEnv>  vec_env_;
    std::unique_ptr<RolloutBuffer>  buffer_;
    std::unique_ptr<StepThreadPool> step_pool_;

    // Carry state — CPU tensors that the next rollout consumes via a single
    // batched H2D transfer at the start of `collect()`.
    torch::Tensor carry_obs_;             // [num_envs, obs_dim]
    torch::Tensor carry_legal_mask_;      // [num_envs, action_count]
    torch::Tensor carry_current_player_;  // [num_envs] int32
    torch::Tensor carry_done_;            // [num_envs]
};

} // namespace poker_ppo
