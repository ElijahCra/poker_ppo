#pragma once

#include "environment.h"
#include "network.h"
#include "types.h"

#include <torch/torch.h>
#include <iosfwd>
#include <string>
#include <vector>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// EloLeague  — lightweight self-play progress tracker
// ─────────────────────────────────────────────────────────────────────────────
//
// Maintains an in-memory pool of ActorCritic snapshots (checkpoints) and
// assigns each a live Elo rating based on the outcomes of round-robin
// self-play matches.  This is far cheaper than exploitability and catches
// regressions in the training run: if your latest checkpoint can't beat the
// ones from 500 updates ago, something is wrong.
//
// Typical usage:
//
//   EloLeague league(factory, bet_cfg, obs_dim, action_count,
//                    hidden_dim, num_layers, EloLeague::Config{});
//
//   // Anchor the initial (untrained) network at rating 1200 so that every
//   // later rating is interpretable as "how much stronger than random".
//   league.add_checkpoint(trainer.network(), 0, 0, "initial");
//   league.set_anchor(0, 1200.0f);
//
//   // Inside the training callback, every K updates:
//   league.add_checkpoint(trainer.network(), update, step, label);
//   league.evaluate_latest();   // play newest vs. all others
//   league.print_standings();
//
// Caveat: head-to-head Elo can rise even in a cycling policy space (the
// intransitive dynamics pointed out in the paper).  Rising Elo is necessary
// for real progress but not sufficient — combine with other diagnostics.

class EloLeague {
public:
    struct Config {
        float initial_rating      = 1200.0f;
        float k_factor            = 32.0f;  // max rating move per match
        int   num_hands_per_match = 1000;   // higher = less noisy, slower
        int   num_parallel_envs   = 32;     // batched inference width
        int   max_checkpoints     = 20;     // prune oldest non-anchored
    };

    struct Checkpoint {
        int         update_idx   = 0;
        int         global_step  = 0;
        float       rating       = 0.0f;
        std::string label;
    };

    struct MatchResult {
        float avg_reward_a = 0.0f;  // mean per-hand reward from A's perspective
        float win_rate_a   = 0.0f;  // (wins + 0.5*ties) / num_hands,  in [0,1]
        int   num_hands    = 0;
    };

    EloLeague(IPokerEnvironmentFactory& factory,
              const BetConfig& bet_cfg,
              int obs_dim,
              int action_count,
              int hidden_dim,
              int num_layers,
              Config cfg,
              torch::Device device = torch::kCPU);

    // ── snapshotting ────────────────────────────────────────────────────

    /// Take a deep copy of `net`'s weights and add it to the pool.
    /// The returned index can be used with `set_anchor`.
    int add_checkpoint(const ActorCritic& net,
                       int update_idx,
                       int global_step,
                       const std::string& label = "");

    /// Fix this checkpoint's rating.  Anchored checkpoints are never
    /// pruned and their rating is never updated by match outcomes.
    void set_anchor(int idx, float rating);

    // ── evaluation ──────────────────────────────────────────────────────

    /// Play round-robin matches between every pair of checkpoints and
    /// update all ratings.  O(N²) matches.
    void run_tournament();

    /// Play the most recently added checkpoint against every existing
    /// checkpoint and update ratings.  O(N) matches.  This is the usual
    /// choice during training.
    void evaluate_latest();

    /// Play a single match between checkpoints i and j.  Does not update
    /// ratings — useful for ad-hoc comparisons.
    MatchResult play_match(int i, int j);

    // ── inspection ──────────────────────────────────────────────────────

    const std::vector<Checkpoint>& checkpoints() const { return ckpts_; }
    void print_standings(std::ostream& os) const;
    void print_standings() const;  // prints to std::cout

private:
    ActorCritic clone_network(const ActorCritic& src);
    MatchResult play_match_internal(ActorCritic& a, ActorCritic& b);
    void        update_ratings(int i, int j, float score_i);

    IPokerEnvironmentFactory& factory_;
    BetConfig                 bet_cfg_;
    Config                    cfg_;
    int                       obs_dim_;
    int                       action_count_;
    int                       hidden_dim_;
    int                       num_layers_;
    torch::Device             device_;

    std::vector<ActorCritic> nets_;       // parallel to ckpts_
    std::vector<Checkpoint>  ckpts_;
    std::vector<bool>        anchored_;
};

}  // namespace poker_ppo
