#pragma once
//
// AllInEquity.hpp — expected showdown equity over board run-outs.
//
// When both players are all-in before the river, the realised winner is
// decided by the (random) cards still to come. Training on that realised
// outcome injects all the run-out variance into the reward; training on
// the *expected* equity at the all-in point removes it. This is the
// standard variance-reduction trick for poker RL.
//
// Equity is exact for the flop (2 cards to come, C(45,2)=990) and turn
// (1 card, 44) — cheap to enumerate. Preflop (5 cards, C(48,5)=1.7M) is
// too many, so it's estimated by Monte-Carlo over `mc_samples` run-outs.
//

#include <cstdint>
#include <random>
#include <utility>

#include "Utility/Utility.hpp"
#include "Utility/CardConversion.hpp"

namespace Game {

// Player-0 equity = P(win) + 0.5·P(tie) over completions of the board.
//   hole0/hole1 : 2 deck-ids each (0..51)
//   board       : n_board revealed community cards (deck-ids); n_board ∈ {0,3,4}
//   rng         : used only for the preflop Monte-Carlo path
//   mc_samples  : run-outs to sample when n_board < 3
// Returns equity0 in [0, 1].
[[nodiscard]] inline double all_in_equity_p0(
    const uint8_t* hole0, const uint8_t* hole1,
    const uint8_t* board, int n_board,
    std::mt19937& rng, int mc_samples)
{
    bool used[52] = {};
    used[hole0[0]] = used[hole0[1]] = true;
    used[hole1[0]] = used[hole1[1]] = true;
    for (int i = 0; i < n_board; ++i) used[board[i]] = true;

    int remaining[52];
    int nrem = 0;
    for (int c = 0; c < 52; ++c)
        if (!used[c]) remaining[nrem++] = c;

    const int need = 5 - n_board;

    // 2p2 card arrays: [hole(2) | board(5)]. Slots [2+n_board .. 6] are
    // filled per run-out; the rest are fixed for the whole enumeration.
    int p0[7], p1[7];
    p0[0] = deck_to_two_plus_two(hole0[0]); p0[1] = deck_to_two_plus_two(hole0[1]);
    p1[0] = deck_to_two_plus_two(hole1[0]); p1[1] = deck_to_two_plus_two(hole1[1]);
    for (int i = 0; i < n_board; ++i) {
        const int t = deck_to_two_plus_two(board[i]);
        p0[2 + i] = t;
        p1[2 + i] = t;
    }

    long win0 = 0, win1 = 0, tie = 0, total = 0;
    auto place = [&](int slot, int deck_card) {
        const int t = deck_to_two_plus_two(static_cast<uint8_t>(deck_card));
        p0[2 + n_board + slot] = t;
        p1[2 + n_board + slot] = t;
    };
    auto tally = [&]() {
        const int w = Utility::getWinner(p0, p1);  // 0, 1, or 3 (tie)
        if      (w == 0) ++win0;
        else if (w == 1) ++win1;
        else             ++tie;
        ++total;
    };

    if (need <= 2) {
        // Exact enumeration (flop/turn all-ins; river all-in has need==0).
        if (need == 0) {
            tally();
        } else if (need == 1) {
            for (int a = 0; a < nrem; ++a) { place(0, remaining[a]); tally(); }
        } else {  // need == 2
            for (int a = 0; a < nrem; ++a) {
                place(0, remaining[a]);
                for (int b = a + 1; b < nrem; ++b) { place(1, remaining[b]); tally(); }
            }
        }
    } else {
        // Preflop: Monte-Carlo. `scratch` stays a permutation of the
        // remaining deck across samples; need partial-shuffle swaps draw a
        // uniform distinct run-out each time.
        int scratch[52];
        for (int i = 0; i < nrem; ++i) scratch[i] = remaining[i];
        for (int s = 0; s < mc_samples; ++s) {
            for (int k = 0; k < need; ++k) {
                std::uniform_int_distribution<int> d(k, nrem - 1);
                std::swap(scratch[k], scratch[d(rng)]);
                place(k, scratch[k]);
            }
            tally();
        }
    }

    if (total == 0) return 0.5;
    return (static_cast<double>(win0) + 0.5 * static_cast<double>(tie))
           / static_cast<double>(total);
}

}  // namespace Game
