#pragma once
//
// league.h — anchor-relative progress tracking. Plays the trained policy
// against a fixed set of "anchor" baselines and reports BB/hand per
// match-up. See add_default_anchors() for the built-in anchor set.
//

#include "config.h"
#include "environment.h"
#include "network.h"
#include "policy.h"

#include <torch/torch.h>

#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace poker_ppo {

class League {
public:
    struct Config {
        int   num_hands_per_match = 1000;   // higher = less noisy bb/hand
        int   num_parallel_envs   = 32;     // batched-inference width
        // Reward in this env is scaled by reward_norm = 10 * big_blind, so a
        // scaled reward of 0.1 is exactly 1 BB/hand.  Keep this in sync if you
        // change reward_norm in the env.
        float bb_per_unit_reward  = 10.0f;
    };

    struct MatchResult {
        std::string anchor_name;
        int   num_hands     = 0;
        float avg_reward_a  = 0.0f;   // mean *scaled* per-hand reward (env units)
        float bb_per_hand_a = 0.0f;   // = avg_reward_a * cfg.bb_per_unit_reward
        float win_rate_a    = 0.0f;   // (wins + 0.5*ties) / num_hands
        // Histogram of the trained policy's chosen actions across the match —
        // useful as a mode-collapse diagnostic.
        std::vector<int64_t> action_counts_a;
    };

    League(IPokerEnvironmentFactory& factory,
           const BetConfig&    bet_cfg,
           int                 obs_dim,
           int                 action_count,
           int                 hidden_dim,
           int                 num_layers,
           BetHistoryConfig    hist,
           RoundSummaryConfig  round_summary,
           Config              cfg,
           torch::Device       device = torch::kCPU);

    // ── anchor registration ─────────────────────────────────────────────

    /// Register all five built-in anchor policies (uniform, random_init,
    /// always_call, always_raise, pair_caller).
    void add_default_anchors();

    /// Register a user-supplied anchor.  Takes ownership.
    void add_anchor(std::unique_ptr<IPolicy> policy);

    /// View into the registered anchors (for diagnostics / printing).
    const std::vector<std::unique_ptr<IPolicy>>& anchors() const {
        return anchors_;
    }

    // ── evaluation ──────────────────────────────────────────────────────

    /// Play `trained` against every registered anchor and return one
    /// MatchResult per anchor (in registration order).
    std::vector<MatchResult> evaluate(const ActorCritic& trained);

    /// Play `a` vs `b` head-to-head.  Used by `evaluate()` and exposed for
    /// ad-hoc comparisons.  Stats are reported from policy A's perspective.
    MatchResult play_match(IPolicy& a, IPolicy& b);

    // ── pretty-print ────────────────────────────────────────────────────
    void print_results(const std::vector<MatchResult>& rs,
                       std::ostream& os) const;
    void print_results(const std::vector<MatchResult>& rs) const;

private:
    /// Build an ActorCritic with the same architecture as the trainer's,
    /// loaded with `src`'s weights.  Used to snapshot the trained model
    /// before passing it into the match loop (so further training updates
    /// don't race with evaluation).
    ActorCritic clone_network(const ActorCritic& src);

    /// Build an untrained ActorCritic with random orthogonal init — the
    /// "random_init" anchor.
    ActorCritic make_random_network();

    IPokerEnvironmentFactory& factory_;
    BetConfig                 bet_cfg_;
    Config                    cfg_;
    int                       obs_dim_;
    int                       action_count_;
    int                       hidden_dim_;
    int                       num_layers_;
    BetHistoryConfig          hist_;
    RoundSummaryConfig        round_summary_;
    torch::Device             device_;

    std::vector<std::unique_ptr<IPolicy>> anchors_;
};

}  // namespace poker_ppo
