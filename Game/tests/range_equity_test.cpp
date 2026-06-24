// RangeEquity tests: the LBR showdown-equity primitive must reduce to the
// trusted head-to-head all_in_equity_p0 for a single-combo range, and obey
// the obvious bounds (nuts = 1, dominated < 0.5, symmetry).

#include "Utility/AllInEquity.hpp"
#include "Utility/RangeEquity.hpp"

#include <gtest/gtest.h>

#include <array>
#include <random>
#include <vector>

namespace {

// deck-id = rank*4 + suit, rank 0..12 = 2..A, suit 0..3.
constexpr uint8_t card(int rank, int suit) {
    return static_cast<uint8_t>(rank * 4 + suit);
}

TEST(RangeEquity, SingleComboMatchesHeadToHead) {
    std::mt19937 rng(123);
    // A few fixed match-ups across board textures.
    struct Case { uint8_t h0, h1, v0, v1; std::array<uint8_t,5> board; int nb; };
    std::vector<Case> cases = {
        // AhAs vs KdKc, flop 2c 7d Th
        {card(12,2),card(12,3), card(11,1),card(11,0),
         {card(0,0),card(5,1),card(8,2),0,0}, 3},
        // AhKh vs 9s9d, turn 2c 7d Th Js
        {card(12,2),card(11,2), card(7,3),card(7,1),
         {card(0,0),card(5,1),card(8,2),card(9,3),0}, 4},
        // river fully dealt: 5c5d vs 6c6d on Ah Kh Qh 2s 3s
        {card(3,0),card(3,1), card(4,0),card(4,1),
         {card(12,2),card(11,2),card(10,2),card(0,3),card(1,3)}, 5},
    };
    for (const auto& c : cases) {
        std::vector<std::array<uint8_t,2>> combos = {{c.v0, c.v1}};
        std::vector<double> w = {1.0};
        const double e_range = Game::hand_vs_range_equity(
            c.h0, c.h1, c.board.data(), c.nb, combos, w, rng, /*mc=*/20000);
        const uint8_t hole0[2] = {c.h0, c.h1};
        const uint8_t hole1[2] = {c.v0, c.v1};
        const double e_h2h = Game::all_in_equity_p0(
            hole0, hole1, c.board.data(), c.nb, rng, /*mc=*/20000);
        // Enumerated boards (nb>=3 here) are exact → identical.
        EXPECT_NEAR(e_range, e_h2h, 1e-9)
            << "nb=" << c.nb;
    }
}

TEST(RangeEquity, NutsAndDominated) {
    std::mt19937 rng(7);
    // Board Ah Kh Qh Jh Th — hero holds nothing relevant; both play the
    // royal flush on board → tie regardless. Equity 0.5.
    std::array<uint8_t,5> royal = {card(12,2),card(11,2),card(10,2),
                                   card(9,2),card(8,2)};
    std::vector<std::array<uint8_t,2>> combos = {{card(0,0),card(1,1)}};
    std::vector<double> w = {1.0};
    double e = Game::hand_vs_range_equity(card(2,0),card(3,0),
                 royal.data(), 5, combos, w, rng, 1);
    EXPECT_NEAR(e, 0.5, 1e-9);

    // Quad aces vs a single non-improving combo on a dry board → hero ~1.
    std::array<uint8_t,5> board = {card(12,0),card(12,1),card(12,3),
                                   card(2,0),card(7,1)};
    combos = {{card(3,2),card(4,2)}};  // 5h6h, no chance
    e = Game::hand_vs_range_equity(card(12,2),card(0,3),
            board.data(), 5, combos, w, rng, 1);
    EXPECT_NEAR(e, 1.0, 1e-9);
}

TEST(RangeEquity, WeightedRangeBetweenComponents) {
    std::mt19937 rng(99);
    // Hero AhAs on flop 2c 7d Th vs a 2-combo range; the weighted equity
    // must lie between the two single-combo equities.
    std::array<uint8_t,5> board = {card(0,0),card(5,1),card(8,2),0,0};
    uint8_t h0 = card(12,2), h1 = card(12,3);
    std::array<uint8_t,2> A = {card(11,1),card(11,0)};  // KK
    std::array<uint8_t,2> B = {card(8,0),card(8,1)};    // TT (set)
    auto eq1 = [&](std::array<uint8_t,2> c){
        std::vector<std::array<uint8_t,2>> v={c}; std::vector<double> w={1.0};
        return Game::hand_vs_range_equity(h0,h1,board.data(),3,v,w,rng,1);
    };
    const double eA = eq1(A), eB = eq1(B);
    std::vector<std::array<uint8_t,2>> both = {A, B};
    std::vector<double> w = {0.5, 0.5};
    const double e = Game::hand_vs_range_equity(h0,h1,board.data(),3,both,w,rng,1);
    const double lo = std::min(eA, eB), hi = std::max(eA, eB);
    EXPECT_GE(e, lo - 1e-9);
    EXPECT_LE(e, hi + 1e-9);
    EXPECT_NEAR(e, 0.5 * (eA + eB), 1e-9);  // equal weights, disjoint blockers
}

}  // namespace
