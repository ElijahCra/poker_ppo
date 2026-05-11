#pragma once

#include "types.h"
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace poker_ppo {

// Two-player zero-sum self-play poker env. One hand per instance.
// Obs is from the acting player's perspective; reward is in player 1's
// frame (player 2 negates). Action 0=Fold, 1=Check/Call, 2..2+N=raises;
// raises mask out once max_bets_per_round is reached.
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

// N independent envs stepped lockstep on one thread. Game step is cheap
// enough that batched inference dominates.
class VectorizedEnv {
public:
    VectorizedEnv(IPokerEnvironmentFactory& factory,
                  const BetConfig& cfg,
                  int num_envs);

    int  num_envs()     const { return static_cast<int>(envs_.size()); }
    int  obs_dim()      const { return envs_[0]->obs_dim(); }
    int  action_count() const { return envs_[0]->bet_config().action_count(); }

    torch::Tensor reset_all();

    struct BatchStepResult {
        torch::Tensor observations;       // [N, obs_dim]
        torch::Tensor rewards;            // [N]
        torch::Tensor dones;              // [N]  float (0/1)
        torch::Tensor legal_action_masks; // [N, action_count]
        torch::Tensor current_players;    // [N]  int (0 or 1)
    };
    BatchStepResult step(const std::vector<int>& actions);

    IPokerEnvironment& env(int i) { return *envs_[i]; }
    std::vector<std::unique_ptr<IPokerEnvironment>>& envs_mut() { return envs_; }

private:
    std::vector<std::unique_ptr<IPokerEnvironment>> envs_;

    // Pre-allocated CPU buffers, reused every step.
    torch::Tensor obs_buf_;
    torch::Tensor rewards_buf_;
    torch::Tensor dones_buf_;
    torch::Tensor masks_buf_;
    torch::Tensor players_buf_;
};

} // namespace poker_ppo
