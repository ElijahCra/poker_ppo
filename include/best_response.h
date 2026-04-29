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

#include "environment.h"
#include "network.h"
#include "rollout_buffer.h"
#include "types.h"

#include <torch/torch.h>

#include <memory>
#include <random>
#include <vector>

namespace poker_ppo {

struct BestResponseConfig {
    bool  enabled            = false;  // master toggle
    int   eval_every         = 1000;   // run every N main-trainer updates
    int   updates_per_eval   = 200;    // PPO updates of exploiter per evaluation

    int   num_envs           = 32;
    int   num_steps          = 128;
    int   update_epochs      = 4;
    int   num_minibatches    = 4;

    float learning_rate      = 3.0e-4f;
    float ent_coef           = 0.01f;
    float vf_coef            = 0.5f;
    float clip_coef          = 0.2f;
    float max_grad_norm      = 0.5f;
    float gamma              = 1.0f;
    float gae_lambda         = 1.0f;
    bool  norm_advantages    = true;
    bool  clip_vloss         = false;

    bool  warm_start         = true;

    // Number of fresh exploiter seeds per evaluate() call. When >1, each
    // call trains that many independent exploiters from random init against
    // the same frozen target, and reports max-bb/hand as the canonical
    // lower bound (mean/min/std are also recorded as diagnostics). Each
    // seed's per-eval bb/hand is a valid lower bound on its own; the max
    // over seeds is the tightest bound the budget can produce, and is much
    // less noisy than any single seed. Cost scales linearly. When >1,
    // warm_start is ignored — every seed is fresh.
    int   num_exploiter_seeds = 3;

    // Hands played in the post-training eval-only match between the trained
    // exploiter and the frozen target. The eval-match's bb/hand is the
    // reported BR estimate — using rewards collected during the exploiter's
    // training rollouts would bias the bound downward (the early-training
    // exploiter plays badly while it's still learning). Set to 0 to skip
    // the eval and fall back to training-time reward averaging (debug only).
    int   eval_hands         = 5000;

    float bb_per_unit_reward = 10.0f;  // matches PokerEnvironment::reward_norm
    uint64_t seed            = 0;
};

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
    ActorCritic clone_network(const ActorCritic& src);

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
