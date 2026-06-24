#pragma once
//
// RangeEquity.hpp — equity of one known hand against a WEIGHTED distribution
// of opponent holdings, at the current board, rolled out to showdown.
//
// This is the showdown-equity primitive Local Best Response needs: LBR holds
// known cards and maintains a belief (weighted combo list) over the target's
// hand; to price a call it needs P(LBR wins | checkdown) against that belief.
//
// Single board enumeration: each remaining-board completion ranks LBR's
// 7-card hand ONCE, then compares against every range combo — O(completions ×
// |range|) hand lookups, vs the O(|range|) separate enumerations a per-combo
// all_in_equity_p0 loop would cost. Reduces bit-for-bit to all_in_equity_p0
// for a single-combo range (unit-tested).
//

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "Utility/Utility.hpp"
#include "Utility/CardConversion.hpp"

namespace Game {

// hero hole = {h0,h1}; board = n_board revealed community deck-ids (∈{0,3,4,5}).
// range_combos[i] = villain's 2 deck-ids, range_weights[i] ≥ 0 its belief mass.
// Combos that collide with hero/board are assumed already pruned by the caller;
// combos colliding with a sampled completion card are skipped for that
// completion (standard LBR handling — no per-completion renormalisation).
// Returns hero equity = (Σ win·w + 0.5 Σ tie·w) / Σ decided·w  ∈ [0,1].
// need = 5 − n_board cards to come; enumerated when ≤ 2, else Monte-Carlo.
[[nodiscard]] inline double hand_vs_range_equity(
    uint8_t h0, uint8_t h1,
    const uint8_t* board, int n_board,
    const std::vector<std::array<uint8_t, 2>>& range_combos,
    const std::vector<double>&                 range_weights,
    std::mt19937& rng, int mc_samples)
{
    if (range_combos.empty()) return 0.5;

    bool used_base[52] = {};
    used_base[h0] = used_base[h1] = true;
    for (int i = 0; i < n_board; ++i) used_base[board[i]] = true;

    int remaining[52];
    int nrem = 0;
    for (int c = 0; c < 52; ++c)
        if (!used_base[c]) remaining[nrem++] = c;

    const int need = 5 - n_board;

    // 2p2 id scratch: [hero hole(2) | board(5)] and [villain hole(2) | board(5)].
    int hero[7], vil[7];
    hero[0] = deck_to_two_plus_two(h0);
    hero[1] = deck_to_two_plus_two(h1);
    for (int i = 0; i < n_board; ++i)
        hero[2 + i] = deck_to_two_plus_two(board[i]);

    // Precompute 2p2 ids of each range combo once.
    const size_t R = range_combos.size();
    std::vector<int> vil0(R), vil1(R);
    for (size_t i = 0; i < R; ++i) {
        vil0[i] = deck_to_two_plus_two(range_combos[i][0]);
        vil1[i] = deck_to_two_plus_two(range_combos[i][1]);
    }

    double win = 0.0, tie = 0.0, decided = 0.0;
    int completion[5];  // deck-ids of the drawn board cards

    auto tally = [&]() {
        // Fill the shared board tail (2p2 ids) for this completion.
        for (int k = 0; k < need; ++k)
            hero[2 + n_board + k] = deck_to_two_plus_two(
                static_cast<uint8_t>(completion[k]));
        const int hero_val = Utility::LookupHandValue(hero);

        for (size_t i = 0; i < R; ++i) {
            const uint8_t c0 = range_combos[i][0];
            const uint8_t c1 = range_combos[i][1];
            // Skip combos that collide with this completion's board cards.
            bool collide = false;
            for (int k = 0; k < need; ++k)
                if (completion[k] == c0 || completion[k] == c1) { collide = true; break; }
            if (collide) continue;

            const double w = range_weights[i];
            if (w <= 0.0) continue;

            vil[0] = vil0[i];
            vil[1] = vil1[i];
            for (int k = 2; k < 2 + n_board; ++k) vil[k] = hero[k];
            for (int k = 0; k < need; ++k)
                vil[2 + n_board + k] = hero[2 + n_board + k];

            const int vil_val = Utility::LookupHandValue(vil);
            if      (hero_val > vil_val) win += w;
            else if (hero_val == vil_val) tie += w;
            decided += w;
        }
    };

    if (need == 0) {
        tally();
    } else if (need == 1) {
        for (int a = 0; a < nrem; ++a) { completion[0] = remaining[a]; tally(); }
    } else if (need == 2) {
        for (int a = 0; a < nrem; ++a) {
            completion[0] = remaining[a];
            for (int b = a + 1; b < nrem; ++b) {
                completion[1] = remaining[b];
                tally();
            }
        }
    } else {
        // Preflop / flop-with-3-to-come: Monte-Carlo board completions.
        int scratch[52];
        for (int i = 0; i < nrem; ++i) scratch[i] = remaining[i];
        for (int s = 0; s < mc_samples; ++s) {
            for (int k = 0; k < need; ++k) {
                std::uniform_int_distribution<int> d(k, nrem - 1);
                std::swap(scratch[k], scratch[d(rng)]);
                completion[k] = scratch[k];
            }
            tally();
        }
    }

    if (decided <= 0.0) return 0.5;
    return (win + 0.5 * tie) / decided;
}

}  // namespace Game
