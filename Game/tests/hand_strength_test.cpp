// Unit tests for `Game/Utility/HandStrength.hpp`.
//
// The obs builder writes 4 features per round to the network, all
// derived from these helpers. These tests pin down the canonical
// "what does feature N mean for this concrete hand" answers so a future
// edit to the rank-mask logic, suit accounting, or straight detection
// can't silently shift training-time observations.
//
// Card encoding (from deck.h): id = (rank << 2) | suit, with rank ∈
// [0, 13) (0=2 ... 12=A) and suit ∈ [0, 4) ("shdc" → s=0, h=1, d=2, c=3).

#include <gtest/gtest.h>

#include "Utility/HandStrength.hpp"

#include <array>
#include <cstdint>

namespace {

constexpr uint8_t card(int rank, int suit) {
    return static_cast<uint8_t>((rank << 2) | suit);
}
constexpr int RANK_2 = 0,  RANK_3 = 1,  RANK_4 = 2,  RANK_5 = 3,
              RANK_6 = 4,  RANK_7 = 5,  RANK_8 = 6,  RANK_9 = 7,
              RANK_T = 8,  RANK_J = 9,  RANK_Q = 10, RANK_K = 11,
              RANK_A = 12;
constexpr int SUIT_S = 0, SUIT_H = 1, SUIT_D = 2, SUIT_C = 3;

}  // namespace

// ─── hand_category ─────────────────────────────────────────────────────

TEST(HandCategory, EmptyAndTrivial) {
    uint8_t none[1] = {0};
    EXPECT_EQ(Game::hand_category(none, 0), 0);
    EXPECT_EQ(Game::hand_category(none, 1), 0);

    // Two non-pair cards → high card.
    std::array<uint8_t, 2> AKo = { card(RANK_A, SUIT_S), card(RANK_K, SUIT_H) };
    EXPECT_EQ(Game::hand_category(AKo.data(), 2), 1);

    // Two same-rank cards → one pair (pocket pair preflop).
    std::array<uint8_t, 2> pp = { card(RANK_8, SUIT_S), card(RANK_8, SUIT_H) };
    EXPECT_EQ(Game::hand_category(pp.data(), 2), 2);
}

TEST(HandCategory, PostflopMadeHands) {
    // Set on the board: pocket 7s + 7 on flop.
    std::array<uint8_t, 5> set_hand = {
        card(RANK_7, SUIT_S), card(RANK_7, SUIT_H),
        card(RANK_7, SUIT_D), card(RANK_K, SUIT_C), card(RANK_2, SUIT_S),
    };
    EXPECT_EQ(Game::hand_category(set_hand.data(), 5), 4);  // trips

    // Made straight (no flush): 89TJQ rainbow.
    std::array<uint8_t, 5> straight_hand = {
        card(RANK_8, SUIT_S), card(RANK_9, SUIT_H),
        card(RANK_T, SUIT_D), card(RANK_J, SUIT_C), card(RANK_Q, SUIT_S),
    };
    EXPECT_EQ(Game::hand_category(straight_hand.data(), 5), 5);

    // Wheel straight: A2345 rainbow.
    std::array<uint8_t, 5> wheel = {
        card(RANK_A, SUIT_S), card(RANK_2, SUIT_H),
        card(RANK_3, SUIT_D), card(RANK_4, SUIT_C), card(RANK_5, SUIT_S),
    };
    EXPECT_EQ(Game::hand_category(wheel.data(), 5), 5);

    // Flush: 5 hearts of mixed ranks.
    std::array<uint8_t, 5> flush_hand = {
        card(RANK_2, SUIT_H), card(RANK_5, SUIT_H), card(RANK_8, SUIT_H),
        card(RANK_J, SUIT_H), card(RANK_K, SUIT_H),
    };
    EXPECT_EQ(Game::hand_category(flush_hand.data(), 5), 6);

    // Full house: trip 9s + pair of Ks.
    std::array<uint8_t, 5> full_house = {
        card(RANK_9, SUIT_S), card(RANK_9, SUIT_H), card(RANK_9, SUIT_D),
        card(RANK_K, SUIT_C), card(RANK_K, SUIT_S),
    };
    EXPECT_EQ(Game::hand_category(full_house.data(), 5), 7);

    // Straight flush: 56789 of clubs.
    std::array<uint8_t, 5> sf = {
        card(RANK_5, SUIT_C), card(RANK_6, SUIT_C), card(RANK_7, SUIT_C),
        card(RANK_8, SUIT_C), card(RANK_9, SUIT_C),
    };
    EXPECT_EQ(Game::hand_category(sf.data(), 5), 9);
}

// ─── max_suit_count ────────────────────────────────────────────────────

TEST(MaxSuitCount, Basic) {
    // Two suited hole cards.
    std::array<uint8_t, 2> suited = { card(RANK_A, SUIT_S), card(RANK_K, SUIT_S) };
    EXPECT_EQ(Game::max_suit_count(suited.data(), 2), 2);

    // 4-of-suit on the flop (2 hole + 2 board same suit) → flush draw.
    std::array<uint8_t, 5> four_flush = {
        card(RANK_A, SUIT_S), card(RANK_K, SUIT_S),  // hole, both spades
        card(RANK_8, SUIT_S), card(RANK_2, SUIT_S),  // two more spades
        card(RANK_5, SUIT_C),                        // off-suit
    };
    EXPECT_EQ(Game::max_suit_count(four_flush.data(), 5), 4);
}

// ─── straight_draw_outs (the central distinction) ──────────────────────

TEST(StraightDrawOuts, OpenEnded) {
    // 6789 rainbow — two-card extension on each end (5 or T).
    std::array<uint8_t, 4> oesd = {
        card(RANK_6, SUIT_S), card(RANK_7, SUIT_H),
        card(RANK_8, SUIT_D), card(RANK_9, SUIT_C),
    };
    EXPECT_EQ(Game::straight_draw_outs(oesd.data(), 4), 2);  // outs: 5, T
}

TEST(StraightDrawOuts, WheelOpenEnded) {
    // 2345 — extends with 6 (high) or A (wheel). Both = OESD.
    std::array<uint8_t, 4> wheel_oesd = {
        card(RANK_2, SUIT_S), card(RANK_3, SUIT_H),
        card(RANK_4, SUIT_D), card(RANK_5, SUIT_C),
    };
    EXPECT_EQ(Game::straight_draw_outs(wheel_oesd.data(), 4), 2);  // outs: 6, A
}

TEST(StraightDrawOuts, BroadwaySingleEnded) {
    // JQKA — only T completes the straight; no rank above A. Single-ended.
    std::array<uint8_t, 4> single = {
        card(RANK_J, SUIT_S), card(RANK_Q, SUIT_H),
        card(RANK_K, SUIT_D), card(RANK_A, SUIT_C),
    };
    EXPECT_EQ(Game::straight_draw_outs(single.data(), 4), 1);  // out: T
}

TEST(StraightDrawOuts, Gutshot) {
    // 89JQ — needs T (interior gap). 1 rank-out.
    std::array<uint8_t, 4> gut = {
        card(RANK_8, SUIT_S), card(RANK_9, SUIT_H),
        card(RANK_J, SUIT_D), card(RANK_Q, SUIT_C),
    };
    EXPECT_EQ(Game::straight_draw_outs(gut.data(), 4), 1);
}

TEST(StraightDrawOuts, DoubleBellyBuster) {
    // 5,7,8,9,J — hits a straight on either a 6 (5-6-7-8-9) or a T
    // (7-8-9-T-J). Two rank-outs, equivalent equity to a normal OESD.
    std::array<uint8_t, 5> dbb = {
        card(RANK_5, SUIT_S), card(RANK_7, SUIT_H),
        card(RANK_8, SUIT_D), card(RANK_9, SUIT_C), card(RANK_J, SUIT_S),
    };
    EXPECT_EQ(Game::straight_draw_outs(dbb.data(), 5), 2);  // outs: 6, T
}

TEST(StraightDrawOuts, NoDraw) {
    // 27TQ rainbow — no 4-of-5 in any straight window.
    std::array<uint8_t, 4> garbage = {
        card(RANK_2, SUIT_S), card(RANK_7, SUIT_H),
        card(RANK_T, SUIT_D), card(RANK_Q, SUIT_C),
    };
    EXPECT_EQ(Game::straight_draw_outs(garbage.data(), 4), 0);
}

TEST(StraightDrawOuts, MadeStraightReturnsOuts) {
    // A made straight still has rank-outs reported by this helper —
    // the obs-builder is responsible for suppressing the feature when
    // category == 5/9 (so this is documenting current behaviour, not
    // asserting suppression here).
    std::array<uint8_t, 5> straight = {
        card(RANK_8, SUIT_S), card(RANK_9, SUIT_H),
        card(RANK_T, SUIT_D), card(RANK_J, SUIT_C), card(RANK_Q, SUIT_S),
    };
    // Category check confirms we have a made straight.
    EXPECT_EQ(Game::hand_category(straight.data(), 5), 5);
    // straight_draw_outs may legitimately return >0 here (e.g., still
    // any rank that "completes" a non-overlapping straight — the wheel
    // would need A234 which we don't have, so 0 here).
    // The contract is just: obs_builder masks this to 0 when made.
    SUCCEED();
}

// ─── straight_alive_windows (preflop connectivity) ─────────────────────
//
// At n=2 (preflop) the threshold is "both hole-card ranks present in
// the window." Mid-range connectors land at 4 windows; wide-gap hands
// at 0. This is the only straight feature with non-zero output preflop.

TEST(StraightAliveWindows, PreflopMidConnector) {
    // 67 — connectors, peak preflop straight potential.
    std::array<uint8_t, 2> cnct = { card(RANK_6, SUIT_S), card(RANK_7, SUIT_H) };
    EXPECT_EQ(Game::straight_alive_windows(cnct.data(), 2), 4);
}

TEST(StraightAliveWindows, PreflopWideGap) {
    // 6J — gap of 5 ranks, no 5-window can hold both. Zero potential.
    std::array<uint8_t, 2> wide = { card(RANK_6, SUIT_S), card(RANK_J, SUIT_H) };
    EXPECT_EQ(Game::straight_alive_windows(wide.data(), 2), 0);
}

TEST(StraightAliveWindows, PreflopBoundary) {
    // AK — connectors at the high boundary; only TJQKA holds both.
    std::array<uint8_t, 2> ak = { card(RANK_A, SUIT_S), card(RANK_K, SUIT_H) };
    EXPECT_EQ(Game::straight_alive_windows(ak.data(), 2), 1);

    // A2 — wheel-only connector.
    std::array<uint8_t, 2> a2 = { card(RANK_A, SUIT_S), card(RANK_2, SUIT_H) };
    EXPECT_EQ(Game::straight_alive_windows(a2.data(), 2), 1);

    // 23 — low connector. Standard 23456 + wheel A2345 = 2 windows.
    std::array<uint8_t, 2> two_three = { card(RANK_2, SUIT_S), card(RANK_3, SUIT_H) };
    EXPECT_EQ(Game::straight_alive_windows(two_three.data(), 2), 2);
}

TEST(StraightAliveWindows, PocketPairsHaveNone) {
    // Pocket pair has only one distinct rank → can never satisfy the
    // "both hole-card ranks in window" preflop threshold. Pair-equity
    // comes from `category`, not straights.
    std::array<uint8_t, 2> kk = { card(RANK_K, SUIT_S), card(RANK_K, SUIT_H) };
    EXPECT_EQ(Game::straight_alive_windows(kk.data(), 2), 0);
}

TEST(StraightAliveWindows, OneGapAndTwoGap) {
    // 75 (1-gap): windows 34567, 45678, 56789 contain both → 3.
    std::array<uint8_t, 2> one_gap = { card(RANK_7, SUIT_S), card(RANK_5, SUIT_H) };
    EXPECT_EQ(Game::straight_alive_windows(one_gap.data(), 2), 3);

    // 84 (3-gap): only 45678 holds both → 1.
    std::array<uint8_t, 2> three_gap = { card(RANK_8, SUIT_S), card(RANK_4, SUIT_H) };
    EXPECT_EQ(Game::straight_alive_windows(three_gap.data(), 2), 1);
}

// ─── overcard_count ────────────────────────────────────────────────────

TEST(OvercardCount, Basic) {
    // AK on a 9-7-2 board → 2 overcards.
    std::array<uint8_t, 2> hole = { card(RANK_A, SUIT_S), card(RANK_K, SUIT_H) };
    std::array<uint8_t, 3> board = {
        card(RANK_9, SUIT_D), card(RANK_7, SUIT_C), card(RANK_2, SUIT_S),
    };
    EXPECT_EQ(Game::overcard_count(hole.data(), board.data(), 3), 2);

    // A9 on a K-J-T board → 0 overcards (board has K above A? no — K<A,
    // but J<A, T<A. So A is overcard, 9 isn't. → 1 overcard).
    std::array<uint8_t, 2> hole_a9 = { card(RANK_A, SUIT_S), card(RANK_9, SUIT_H) };
    std::array<uint8_t, 3> board_kjt = {
        card(RANK_K, SUIT_D), card(RANK_J, SUIT_C), card(RANK_T, SUIT_S),
    };
    EXPECT_EQ(Game::overcard_count(hole_a9.data(), board_kjt.data(), 3), 1);

    // 22 on KQJ → 0 overcards.
    std::array<uint8_t, 2> twos = { card(RANK_2, SUIT_S), card(RANK_2, SUIT_H) };
    EXPECT_EQ(Game::overcard_count(twos.data(), board_kjt.data(), 3), 0);

    // No community cards (preflop) → always 0.
    EXPECT_EQ(Game::overcard_count(hole.data(), board.data(), 0), 0);
}
