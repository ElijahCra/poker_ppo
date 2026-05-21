#pragma once
//
// Anchor-relative progress tracking. Plays the trained policy against a
// fixed set of baseline anchors and reports BB/hand per match-up. See
// add_default_anchors() for the built-in set.
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
        int   num_parallel_envs   = 32;
        // Env reward is scaled by 10 * big_blind, so a scaled reward of
        // 0.1 == 1 BB/hand. Keep in sync with PokerEnvironment::reward_norm.
        float bb_per_unit_reward  = 10.0f;
    };

    struct MatchResult {
        std::string anchor_name;
        int   num_hands     = 0;
        float avg_reward_a  = 0.0f;   // mean scaled per-hand reward
        float bb_per_hand_a = 0.0f;   // = avg_reward_best * bb_per_unit_reward
        float win_rate_a    = 0.0f;
        // Action histogram for the trained policy — mode-collapse diagnostic.
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

    // uniform, random_init, always_call, always_raise, pair_caller.
    void add_default_anchors();

    void add_anchor(std::unique_ptr<IPolicy> policy);

    const std::vector<std::unique_ptr<IPolicy>>& anchors() const {
        return anchors_;
    }

    // One MatchResult per anchor, in registration order.
    std::vector<MatchResult> evaluate(const ActorCritic& trained);

    // Stats are from policy A's perspective.
    MatchResult play_match(IPolicy& a, IPolicy& b);

    void print_results(const std::vector<MatchResult>& rs,
                       std::ostream& os) const;
    void print_results(const std::vector<MatchResult>& rs) const;

private:
    // Snapshot before the match so further training can't race with eval.
    ActorCritic clone_network(const ActorCritic& src);

    // Untrained ActorCritic with random orthogonal init. The "random_init" anchor.
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
