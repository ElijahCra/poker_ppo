// allin_equity_test.cpp — unit tests for all-in showdown equity.
//
// Game::all_in_equity_p0 averages the showdown result over the board
// run-outs still to come (exact for flop/turn, Monte-Carlo preflop). These
// tests pin: certain wins (=1), symmetry/zero-sum (eq0+eq1=1), and that the
// Monte-Carlo preflop estimate lands on well-known equities (AA vs 22).
//
// Card encoding (deck.h): id = (rank << 2) | suit,  suit "shdc" (0=s..3=c),
// rank 0='2' .. 12='A'.

#include "Utility/AllInEquity.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>

namespace {

enum SuitDh : int { S = 0, H = 1, D = 2, C = 3 };
constexpr int R2 = 0, R3 = 1, R4 = 2, R5 = 3, R6 = 4, R7 = 5, R8 = 6;
constexpr int R9 = 7, RT = 8, RJ = 9, RQ = 10, RK = 11, RA = 12;

[[nodiscard]] constexpr uint8_t dh(int rank, SuitDh suit) {
    return static_cast<uint8_t>((rank << 2) | static_cast<int>(suit));
}

double equity0(const std::array<uint8_t, 2>& h0,
               const std::array<uint8_t, 2>& h1,
               const std::vector<uint8_t>& board,
               int mc_samples = 5000,
               uint64_t seed = 0xC0FFEE) {
    std::mt19937 rng(seed);
    return Game::all_in_equity_p0(h0.data(), h1.data(),
                                  board.data(), static_cast<int>(board.size()),
                                  rng, mc_samples);
}

// ─── Certain wins: a made nut hand can't be run down (exact path) ─────────

TEST(AllInEquity, RoyalFlushOnFlopIsCertainWin) {
    // P0 holds As Ks; flop Qs Js Ts → royal flush already made. Two cards to
    // come (exact enumeration over C(45,2)); P0 wins every run-out.
    const std::array<uint8_t, 2> h0{dh(RA, S), dh(RK, S)};
    const std::array<uint8_t, 2> h1{dh(R2, C), dh(R3, C)};
    const std::vector<uint8_t>   board{dh(RQ, S), dh(RJ, S), dh(RT, S)};
    EXPECT_DOUBLE_EQ(equity0(h0, h1, board), 1.0);
}

TEST(AllInEquity, RoyalFlushOnTurnIsCertainWin) {
    // need == 1 (exact over the 44 remaining cards).
    const std::array<uint8_t, 2> h0{dh(RA, S), dh(RK, S)};
    const std::array<uint8_t, 2> h1{dh(R2, C), dh(R3, C)};
    const std::vector<uint8_t>   board{dh(RQ, S), dh(RJ, S), dh(RT, S), dh(R2, H)};
    EXPECT_DOUBLE_EQ(equity0(h0, h1, board), 1.0);
}

// ─── Symmetry: swapping the two hands must give complementary equity ──────

TEST(AllInEquity, SymmetricOnFlop) {
    const std::array<uint8_t, 2> h0{dh(RA, S), dh(RK, H)};
    const std::array<uint8_t, 2> h1{dh(RQ, D), dh(RQ, C)};
    const std::vector<uint8_t>   board{dh(R2, S), dh(R7, H), dh(RT, D)};
    const double e0 = equity0(h0, h1, board);
    const double e1 = equity0(h1, h0, board);
    EXPECT_NEAR(e0 + e1, 1.0, 1e-9);   // exact path: ties split, sums to 1
}

// ─── Monte-Carlo preflop estimate matches the textbook equity ─────────────

TEST(AllInEquity, PocketAcesVsPocketDeucesPreflop) {
    // AA vs 22 is ≈ 0.82 for the overpair.
    const std::array<uint8_t, 2> aces{dh(RA, S), dh(RA, H)};
    const std::array<uint8_t, 2> deuces{dh(R2, C), dh(R2, D)};
    const double e = equity0(aces, deuces, /*board=*/{}, /*mc=*/20000);
    EXPECT_NEAR(e, 0.82, 0.02);
}

TEST(AllInEquity, TrashVsAcesIsBigUnderdogPreflop) {
    const std::array<uint8_t, 2> trash{dh(R2, C), dh(R7, D)};
    const std::array<uint8_t, 2> aces{dh(RA, S), dh(RA, H)};
    const double e = equity0(trash, aces, /*board=*/{}, /*mc=*/20000);
    EXPECT_LT(e, 0.20);
    EXPECT_GT(e, 0.05);
}

// ─── Determinism: same seed ⇒ identical Monte-Carlo estimate ──────────────

TEST(AllInEquity, MonteCarloIsDeterministicGivenSeed) {
    const std::array<uint8_t, 2> h0{dh(RA, S), dh(RK, S)};
    const std::array<uint8_t, 2> h1{dh(RQ, D), dh(RJ, C)};
    EXPECT_DOUBLE_EQ(equity0(h0, h1, {}, 3000, 123),
                     equity0(h0, h1, {}, 3000, 123));
}

}  // namespace
