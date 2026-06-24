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

namespace poker_ppo {

class PokerEnvironment;

struct LBRConfig {
    int      num_hands         = 20000;
    int      equity_mc_samples = 600;  // board-completion MC when ≥3 to come
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
