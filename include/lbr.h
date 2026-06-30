#pragma once
//
// Local Best Response (Lisý & Bowling 2017) — a training-free exploitability
// lower bound for the target policy. Unlike the PPO best-response evaluator
// (a learned attacker with its own entropy floor / value-head OOD), LBR
// reasons exactly: it keeps a Bayesian belief over the target's hand
// (updated from the target's OWN action probabilities) and, at each of its
// decisions, picks the action with the highest immediate EV under that
// belief.
//
// v1 — action set {fold, call}: LBR never bets, so it needs no model of the
// target's response to a raise; it calls iff its showdown equity vs the
// current belief beats the pot odds, assuming a checkdown afterwards. This
// is the standard tractable LBR variant. It is a WEAKER attacker than full
// LBR (no value-raises / bluffs), so its value is a conservative lower
// bound: if it already beats the target, the target is exploitable; if it
// loses, a raise-enabled LBR (v2) may still win. Either way the number is a
// valid lower bound on exploitability (which is ≥ 0 for a true BR).
//

#include "config.h"
#include "environment.h"
#include "network.h"

#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <random>
#include <string>

namespace poker_ppo {

class PokerEnvironment;

struct LBRConfig {
    int      num_hands         = 20000;
    int      equity_mc_samples = 600;  // board-completion MC when ≥3 to come
    // v2: also consider raising. For each candidate size, the target's
    // fold/call response is read from its own policy at the post-raise node
    // (snapshot → apply → query → restore). false → v1 {fold,call} only.
    bool     enable_raises     = true;
    // Deployment filter applied to the target before attacking it: temper
    // then drop actions below play_min_p and renormalise (matches play
    // mode). 0/1.0 = attack the raw policy. Lets LBR measure the
    // exploitability of the policy as actually DEPLOYED, not just raw.
    float    play_min_p        = 0.0f;
    float    play_temp         = 1.0f;
    // Interpret the target's actor head as RM⁺ regrets (clamp≥0, normalise
    // over legal) instead of a softmax policy — lets LBR attack the CURRENT σ
    // (an ESCHER regret net) rather than an average/softmax policy.
    bool     rm_plus           = false;
    // Optional per-hand diagnostic JSONL (empty = off). The aggregate
    // breakdown table is always printed.
    std::string log_path;
    // Measure the target's fold-rate to a hypothetical pot-sized raise at
    // every LBR decision, bucketed by street — a direct read on whether the
    // bot over-folds to pressure on ALL streets (LBR only realises river
    // raises, so this probe is what exposes flop/turn fold-weakness). Adds
    // one extra target query per LBR node; off by default.
    bool     fold_probe        = false;
    // Raise pricing for streets before the river.
    //   0 = analytic checkdown, river raises only — EXACT, the default and
    //       the trustworthy bound.
    //   1 = MC rollout, river-only — a VALIDATION mode: converges to the
    //       analytic bound as K grows (confirmed: K24→1.43, K120→1.79 vs
    //       analytic 2.22 on the test model), proving the rollout machinery
    //       (belief-sampled holding + consistent-board injection, no
    //       hidden-info leak) is sound.
    //   2 = MC rollout, ALL streets — EXPERIMENTAL, NOT a trustworthy bound.
    //       LBR's rollout policy (commit to call-down after raising) is too
    //       passive on flop/turn: it pays off the target's later value bets,
    //       so the realised value of early raises is negative regardless of
    //       K. A meaningful all-street bound needs recursive optimal
    //       future-street play inside the rollout (large build). The
    //       calibration line flags the resulting EV as OPTIMISTIC.
    int      rollout_mode      = 0;
    int      rollout_samples   = 24;   // K rollouts per candidate raise
    uint64_t seed              = 0;
};

class LBREvaluator {
public:
    LBREvaluator(IPokerEnvironmentFactory& factory,
                 const BetConfig&          bet_cfg,
                 LBRConfig                 cfg,
                 torch::Device             device);

    struct Result {
        int    num_hands    = 0;
        double mbb_per_hand = 0.0;
        double bb_per_hand  = 0.0;   // lower bound on exploitability
        double lbr_win_rate = 0.0;
        double wall_ms      = 0.0;
    };

    // Play cfg.num_hands of LBR vs target (alternating seats). target is
    // attacked as-is (raw policy, no deployment filter).
    Result evaluate(ActorCritic& target);

private:
    IPokerEnvironmentFactory&          factory_;
    BetConfig                          bet_cfg_;
    LBRConfig                          cfg_;
    torch::Device                      device_;
    std::unique_ptr<IPokerEnvironment> env_owned_;
    PokerEnvironment*                  env_ = nullptr;  // non-owning view
    std::mt19937                       rng_;
};

}  // namespace poker_ppo
