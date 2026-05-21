#pragma once

// Approximate best-response (ABR)
// Trains a vanilla PPO exploiter against a frozen version of the trained model
// The exploiter's terminal reward is a valid lower bound on target exploitability
// when num_seeds > 1 trains multiple exploiters for variance reduction / better lower bound

#include "config.h"
#include "environment.h"
#include "network.h"
#include "rollout.h"

#include <torch/torch.h>

#include <memory>
#include <random>
#include <vector>

namespace poker_ppo {

class BestResponseEvaluator {
public:
    struct Result {
        int    update;
        int    global_step;
        int    br_updates_run;

        // Stats from the seed that achieved max bb/hand
        int    num_hands;
        float  avg_reward_best;
        float  bb_per_hand_best;
        float  win_rate_best;

        // Aggregates across all seeds.
        int    num_seeds;
        float  bb_per_hand_mean;
        float  bb_per_hand_min;
        float  bb_per_hand_std;

        double wall_ms;
    };

    BestResponseEvaluator(IPokerEnvironmentFactory& factory,
                          const BetConfig&    bet_cfg,
                          int                 obs_dim,
                          int                 action_count,
                          int                 hidden_dim,
                          int                 num_layers,
                          BetHistoryConfig    hist,
                          RoundSummaryConfig  round_summary,
                          BestResponseConfig  cfg,
                          torch::Device       device);

    // Trains one or multiple exploiters for cfg.updates_per_eval updates
    Result evaluate(const ActorCritic& target, int update, int global_step);

    const BestResponseConfig& config() const { return cfg_; }

private:
    void init_exploiter();

    struct EvalStats {
        int    num_hands    = 0;
        int    wins         = 0;
        int    ties         = 0;
        double total_reward = 0.0;
    };
    EvalStats eval_match(ActorCritic& target);

    // Train one exploiter from current state for cfg_.updates_per_eval updates
    EvalStats run_one_seed(ActorCritic& frozen_target);

    IPokerEnvironmentFactory& factory_;
    BetConfig                 bet_cfg_;
    BestResponseConfig        cfg_;
    int                       obs_dim_;
    int                       action_count_;
    int                       hidden_dim_;
    int                       num_layers_;
    BetHistoryConfig          hist_;
    RoundSummaryConfig        round_summary_;
    torch::Device             device_;

    ActorCritic                                        exploiter_{nullptr};
    std::unique_ptr<torch::optim::Adam>                optimizer_;
    std::unique_ptr<RolloutBuffer>                     buffer_;
    std::vector<std::unique_ptr<IPokerEnvironment>>    envs_;

    bool         warm_started_ = false;
    std::mt19937 rng_;
};

}  // namespace poker_ppo
