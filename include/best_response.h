#pragma once
//
// best_response.h — approximate best response (ABR) evaluator for tracking
// approximate exploitability of the trained policy during training.
//
// Approach: the first paragraph of Section 3 of [Timbers et al. 2020,
// "Approximate Exploitability: Learning a Best Response"] observes that
// fixing the opponent's policy turns the env into a stationary single-agent
// problem; an RL-trained best responder's reward is then a valid lower bound
// on the target's true exploitability. We use vanilla PPO as the responder
// rather than the paper's IS-MCTS-BR variant — the search-based variant
// costs ~100k learning steps per evaluation, while a vanilla PPO exploiter
// trained for K updates (K << 1k) fits inside a 1000-update training-evaluation
// cadence at modest overhead.
//
// Warm-start: the exploiter's weights persist across evaluate() calls by
// default, so each evaluation continues training from where the previous one
// left off, but against a fresh frozen target. Set warm_start = false to
// re-init on every call.
//
// Correctness note: the exploiter is a single-agent learner in the MDP where
// the target is part of the env dynamics. Only the exploiter-seat
// transitions are stored in the rollout buffer; opponent transitions are
// absorbed into the next exploiter-action's accumulated reward. We treat
// rollout-end truncations as terminals (V_next = 0) rather than bootstrapping
// from V(carry_obs), since the exploiter's value head is only trained on
// exploiter-acting states and is OOD on target-acting carry states.
//

#include "config.h"
#include "environment.h"
#include "network.h"
#include "rollout_buffer.h"

#include <torch/torch.h>

#include <memory>
#include <random>
#include <vector>

namespace poker_ppo {

class BestResponseEvaluator {
public:
    struct Result {
        int    update;             // main-trainer update_idx_ at evaluation
        int    global_step;        // main-trainer global_step_
        int    br_updates_run;     // PPO updates per exploiter seed

        // Stats from the seed that achieved max bb/hand — that's the
        // tightest measured lower bound on exploitability.
        int    num_hands;          // hands in the best-seed eval match
        float  avg_reward_a;       // best-seed scaled mean reward
        float  bb_per_hand_a;      // best-seed bb/hand = max over seeds
        float  win_rate_a;         // best-seed (wins + 0.5*ties) / num_hands

        // Per-seed aggregates (across all `num_seeds` exploiters).
        int    num_seeds;          // = cfg.num_exploiter_seeds
        float  bb_per_hand_mean;
        float  bb_per_hand_min;
        float  bb_per_hand_std;    // population std

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

    /// Train a fresh-or-warm-started exploiter against a frozen copy of
    /// `target` for `cfg.updates_per_eval` updates and return the result.
    /// `update` and `global_step` are stamped into the Result for logging.
    Result evaluate(const ActorCritic& target, int update, int global_step);

    const BestResponseConfig& config() const { return cfg_; }

private:
    void init_exploiter();

    // Stats from a deterministic, no-learning match between the exploiter
    // and the frozen target — the canonical BR measurement once training
    // completes. eval_match() resets envs_ before running.
    struct EvalStats {
        int    num_hands    = 0;
        int    wins         = 0;
        int    ties         = 0;
        double total_reward = 0.0;
    };
    EvalStats eval_match(ActorCritic& target);

    // Train one exploiter from current state for `cfg_.updates_per_eval`
    // updates against the frozen target, then run the eval-match. Each
    // seed of evaluate() invokes this once with a fresh exploiter.
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

    // Persistent exploiter network + optimiser (re-init on first call or
    // when warm_start is off).
    ActorCritic                                        exploiter_{nullptr};
    std::unique_ptr<torch::optim::Adam>                optimizer_;
    std::unique_ptr<RolloutBuffer>                     buffer_;
    std::vector<std::unique_ptr<IPokerEnvironment>>    envs_;

    bool         warm_started_ = false;
    std::mt19937 rng_;
};

}  // namespace poker_ppo
