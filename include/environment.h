#pragma once

#include "types.h"
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace poker_ppo {

// Two-player zero-sum self-play poker env. One hand per instance. The
// observation vector is from the acting player's perspective; reward is
// from player 1's perspective (player 2 gets -reward). Action 0=Fold,
// 1=Check/Call, 2..2+N=raises per BetConfig (mask all raises once the
// player hits max_bets_per_round).

class IPokerEnvironment {
public:
    virtual ~IPokerEnvironment() = default;

    virtual int obs_dim() const = 0;
    virtual const BetConfig& bet_config() const = 0;

    virtual StepResult reset() = 0;
    virtual StepResult step(int action) = 0;

    virtual int current_player() const = 0;
    virtual torch::Tensor observation() const = 0;
    virtual torch::Tensor legal_action_mask() const = 0;
    virtual bool is_terminal() const = 0;
};

class IPokerEnvironmentFactory {
public:
    virtual ~IPokerEnvironmentFactory() = default;
    virtual std::unique_ptr<IPokerEnvironment> create(const BetConfig& cfg) = 0;
};

// Owns N independent envs and runs them in lockstep on a single thread —
// these games are tiny enough that batched-inference dominates step time.

class VectorizedEnv {
public:
    VectorizedEnv(IPokerEnvironmentFactory& factory,
                  const BetConfig& cfg,
                  int num_envs);

    int  num_envs()     const { return static_cast<int>(envs_.size()); }
    int  obs_dim()      const { return envs_[0]->obs_dim(); }
    int  action_count() const { return envs_[0]->bet_config().action_count(); }

    /// Reset all environments.  Returns stacked observations [num_envs, obs_dim].
    torch::Tensor reset_all();

    /// Step each environment with its corresponding action.
    /// actions: vector of int, length num_envs.
    /// Returns: observations [N, obs_dim], rewards [N], dones [N], masks [N, A].
    struct BatchStepResult {
        torch::Tensor observations;      // [N, obs_dim]
        torch::Tensor rewards;           // [N]
        torch::Tensor dones;             // [N]  float (0/1)
        torch::Tensor legal_action_masks; // [N, action_count]
        torch::Tensor current_players;   // [N]  int (0 or 1)
    };
    BatchStepResult step(const std::vector<int>& actions);

    /// Access individual envs (e.g. for current_player queries).
    IPokerEnvironment& env(int i) { return *envs_[i]; }

    /// Raw access to the owning vector of envs
    std::vector<std::unique_ptr<IPokerEnvironment>>& envs_mut() { return envs_; }

private:
    std::vector<std::unique_ptr<IPokerEnvironment>> envs_;

    // Pre-allocated CPU output tensors — reused every step to avoid heap allocs
    torch::Tensor obs_buf_;      // [N, obs_dim]
    torch::Tensor rewards_buf_;  // [N]
    torch::Tensor dones_buf_;    // [N]
    torch::Tensor masks_buf_;    // [N, A]
    torch::Tensor players_buf_;  // [N]  int32
};

} // namespace poker_ppo
