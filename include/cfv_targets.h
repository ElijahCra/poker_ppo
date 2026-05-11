#pragma once
//
// Counterfactual-value target computation for the auxiliary CFV head.
//
// At a terminal state, both seats' hole cards and the full board are known
// to the trainer. For a given player, we compute, for each of the C(52,2)
// = 1326 possible hole-card combos:
//
//   target[combo] = hypothetical showdown utility for the player if they
//                   had `combo` instead of their actual hand, holding the
//                   action sequence + opponent hand + board fixed.
//   mask[combo]   = 1 iff `combo` is consistent with opp's actual hand
//                   and the actual board (no card overlaps), else 0.
//
// We always evaluate hypothetical *showdown* values, even if the actual
// terminal was a fold. Strictly biased CFV at fold terminals, but a useful
// auxiliary signal — the encoder learns range-aware features either way.
//
// Output is in raw chip units; caller divides by reward_norm to match the
// training reward scale.
//

#include "config.h"   // for kCFVHeadDim

#include <array>

namespace poker_ppo {

class PokerEnvironment;

struct CFVTerminalResult {
    std::array<float, kCFVHeadDim> target;
    std::array<float, kCFVHeadDim> mask;
};

// Computes (target, mask) for `player` at the terminal state of `env`.
// Caller must ensure env.is_terminal().
CFVTerminalResult compute_cfv_at_terminal(const PokerEnvironment& env,
                                          int                     player);

// Combo encoding: combos are indexed in lex order, combo i = (a, b) where
// 0 <= a < b < 52, and i goes 0..1325. Exposed for rollout-side bookkeeping
// and tests. `combo_to_cards(i)` returns the (a, b) pair.
std::pair<int, int> combo_to_cards(int combo_idx) noexcept;

}  // namespace poker_ppo
