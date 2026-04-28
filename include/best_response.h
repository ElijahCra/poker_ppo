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
    float bb_per_unit_reward = 10.0f;  // matches PokerEnvironment::reward_norm
    uint64_t seed            = 0;
};

class BestResponseEvaluator {
public:
    struct Result {
        int    update;             // main-trainer update_idx_ at evaluation
        int    global_step;        // main-trainer global_step_
        int    br_updates_run;
        int    num_hands;
        float  avg_reward_a;       // exploiter's mean per-hand reward (scaled)
        float  bb_per_hand_a;      // = avg_reward_a * bb_per_unit_reward
        float  win_rate_a;         // (wins + 0.5*ties) / num_hands
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
