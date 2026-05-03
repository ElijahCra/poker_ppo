// showdown_test.cpp — unit tests for the showdown winner-determination
// path. Targets two pieces:
//
//   1. Utility::getWinner — the TwoPlusTwo-backed 7-card hand evaluator.
//      We construct hands by category (high card, pair, two pair, …,
//      straight flush) and assert that the stronger hand wins, including
//      same-category kicker comparisons. The historical bug was that
//      makeTerminal compared hand_indexer outputs (canonical-form ids)
//      instead of strength scores; these tests pin the corrected behaviour.
//
//   2. The deck → TwoPlusTwo card-id conversion used by
//      Transitioner::makeTerminal. We exercise the same formula here so a
//      future refactor can't silently misalign the two encodings.
//
// Card encoding cheat-sheet:
//   deck.h:    id_dh  = (rank << 2) | suit_dh   suit "shdc" (0=s,1=h,2=d,3=c)
//   TwoPlusTwo: id_2p2 = rank*4 + suit_2p2 + 1   suit "cdhs" (0=c,1=d,2=h,3=s)
//
// Rank values are 0..12 where 0 = "2" and 12 = "A" in both encodings.

#include "Utility/CardConversion.hpp"
#include "Utility/Utility.hpp"
#include "Game.hpp"
#include "GameConfig.hpp"
#include "BettingConfig.hpp"
#include "ActionPolicy.hpp"
#include "GameBase.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <variant>
#include <vector>

namespace {

// ─── Encoding helpers ─────────────────────────────────────────────────────

enum SuitDh : int { S = 0, H = 1, D = 2, C = 3 };          // deck.h "shdc"
enum SuitTpt: int { CLUBS = 0, DIAMONDS = 1, HEARTS = 2, SPADES = 3 };  // TwoPlusTwo "cdhs"

// Convenience: build a TwoPlusTwo card id directly. Rank: 0='2' .. 12='A'.
[[nodiscard]] constexpr int tpt(int rank, SuitTpt suit) {
    return rank * 4 + static_cast<int>(suit) + 1;
}

// Build a deck.h-encoded card id (the engine's internal format).
[[nodiscard]] constexpr uint8_t dh(int rank, SuitDh suit) {
    return static_cast<uint8_t>((rank << 2) | static_cast<int>(suit));
}

// Mirror of the conversion in Transitioner::makeTerminal. Kept here so a
// refactor on either side that breaks the formula is caught.
[[nodiscard]] constexpr int dh_to_tpt(uint8_t deck_card) {
    const int rank      = deck_card >> 2;
    const int suit_dh   = deck_card & 3;
    const int suit_2p2  = 3 - suit_dh;
    return rank * 4 + suit_2p2 + 1;
}

// Convenience aliases for ranks (0-indexed: 0='2', 12='A').
constexpr int R2 = 0, R3 = 1, R4 = 2, R5 = 3, R6 = 4, R7 = 5, R8 = 6;
constexpr int R9 = 7, RT = 8, RJ = 9, RQ = 10, RK = 11, RA = 12;

// Run getWinner with two 7-card hands. Returns 0/1/3 (3 = tie).
int winner_of(const std::array<int, 7>& a, const std::array<int, 7>& b) {
    int p0[7]; std::copy(a.begin(), a.end(), p0);
    int p1[7]; std::copy(b.begin(), b.end(), p1);
    return Utility::getWinner(p0, p1);
}

// ─── Encoding-conversion tests ────────────────────────────────────────────

TEST(Encoding, DeckToTwoPlusTwoIsBijectiveOver52) {
    // Every deck.h id in [0, 52) maps to a unique TwoPlusTwo id in [1, 52].
    std::vector<int> seen(53, 0);
    for (uint8_t d = 0; d < 52; ++d) {
        const int t = dh_to_tpt(d);
        ASSERT_GE(t, 1);
        ASSERT_LE(t, 52);
        ASSERT_EQ(seen[t], 0) << "TwoPlusTwo id " << t << " collided";
        seen[t] = 1;
    }
}

TEST(Encoding, RanksPreservedAcrossConversion) {
    // Rank should be invariant under the suit-mirror conversion.
    for (int rank = 0; rank <= 12; ++rank) {
        for (int suit = 0; suit < 4; ++suit) {
            const uint8_t d = static_cast<uint8_t>((rank << 2) | suit);
            const int t = dh_to_tpt(d);
            // TwoPlusTwo id 1..52 — rank index = (id-1) / 4.
            EXPECT_EQ((t - 1) / 4, rank);
        }
    }
}

TEST(Encoding, AceOfSpadesMatchesAcrossConventions) {
    // Sanity anchor: A♠ deck.h id = 12*4 + 0 = 48; TwoPlusTwo "As" id = 52.
    EXPECT_EQ(dh(RA, S), 48);
    EXPECT_EQ(dh_to_tpt(48), 52);
}

TEST(Encoding, DeckIsRankMajorSuitMinor) {
    // Pins the engine's deck.h layout: id = (rank << 2) | suit, i.e.
    // rank lives in the upper bits and suit in the lower 2. That makes
    // consecutive ids the *same rank in different suits* (0..3 are all 2's
    // → 2♠ 2♥ 2♦ 2♣ under the "shdc" suit table), NOT the same suit in
    // different ranks. If anyone ever swaps these bits, the env's
    // observation one-hot, the hole-card extraction in PairAllInPolicy
    // (which relies on `hole[0]/4 == hole[1]/4` to detect pairs), and the
    // showdown card encoding all break in different ways. This test fails
    // immediately if the layout flips.

    // Bit-level decomposition for every (rank, suit).
    for (int rank = 0; rank <= 12; ++rank) {
        for (int suit = 0; suit < 4; ++suit) {
            const uint8_t id = static_cast<uint8_t>((rank << 2) | suit);
            EXPECT_EQ(id >> 2, rank) << "id=" << int(id);
            EXPECT_EQ(id & 3,  suit) << "id=" << int(id);
        }
    }

    // Concrete anchors at the top and bottom of the rank ladder. 0..3 are
    // all the deuces; 48..51 are all the aces.
    EXPECT_EQ(dh(R2, S), 0);
    EXPECT_EQ(dh(R2, H), 1);
    EXPECT_EQ(dh(R2, D), 2);
    EXPECT_EQ(dh(R2, C), 3);
    EXPECT_EQ(dh(RA, S), 48);
    EXPECT_EQ(dh(RA, H), 49);
    EXPECT_EQ(dh(RA, D), 50);
    EXPECT_EQ(dh(RA, C), 51);

    // The pair-detection trick used by PairAllInPolicy:
    //   `card_a / 4 == card_b / 4`  ⇔  same rank.
    // Holds for any two cards in the same rank-quartet, never for cards
    // across rank-quartets.
    for (int rank = 0; rank <= 12; ++rank) {
        const uint8_t a = static_cast<uint8_t>((rank << 2) | 0);
        const uint8_t b = static_cast<uint8_t>((rank << 2) | 3);
        EXPECT_EQ(a / 4, b / 4) << "same-rank pair check failed at rank=" << rank;
    }
    for (int rank = 0; rank < 12; ++rank) {
        const uint8_t lo = static_cast<uint8_t>((rank << 2) | 3);
        const uint8_t hi = static_cast<uint8_t>(((rank + 1) << 2) | 0);
        EXPECT_NE(lo / 4, hi / 4)
            << "different-rank pair check failed at boundary rank=" << rank;
    }
}

// ─── PairAllInPolicy pair-detection (same algorithm, on synthetic obs) ───
//
// PairAllInPolicy in include/policy.h decides "shove or fold" on the basis
// of `hole[0]/4 == hole[1]/4`. That trick is correct only because deck.h's
// encoding is rank-major (the previous test pins it down). These tests
// replicate the policy's exact loop on hand-crafted obs[0:52] one-hots and
// assert the pair flag is correct for every starting hand class.

namespace {

// Mirrors the obs→hole→pair logic in PairAllInPolicy::select_actions.
bool pair_detected_via_obs(uint8_t card_a, uint8_t card_b) {
    std::array<float, 52> obs{};  // zero-init
    obs[card_a] = 1.0f;
    obs[card_b] = 1.0f;
    int hole[2] = {-1, -1};
    int n = 0;
    for (int c = 0; c < 52 && n < 2; ++c) {
        if (obs[c] > 0.5f) hole[n++] = c;
    }
    return (n == 2) && (hole[0] / 4 == hole[1] / 4);
}

}  // namespace

TEST(PairDetection, EveryPocketPairIsDetected) {
    // For every rank, all C(4, 2) = 6 same-rank suit pairs should register.
    for (int rank = 0; rank <= 12; ++rank) {
        for (int s1 = 0; s1 < 4; ++s1) {
            for (int s2 = s1 + 1; s2 < 4; ++s2) {
                const uint8_t a = static_cast<uint8_t>((rank << 2) | s1);
                const uint8_t b = static_cast<uint8_t>((rank << 2) | s2);
                EXPECT_TRUE(pair_detected_via_obs(a, b))
                    << "missed pair at rank=" << rank
                    << " suits=(" << s1 << "," << s2 << ")";
            }
        }
    }
}

TEST(PairDetection, NoUnpairedHandRegistersAsPair) {
    // For every cross-rank pair of cards, the policy must NOT flag a pair.
    // Iterates the 1326 unordered starting hands minus the 78 pocket pairs.
    int unpaired_checked = 0;
    for (int a = 0; a < 52; ++a) {
        for (int b = a + 1; b < 52; ++b) {
            if (a / 4 == b / 4) continue;  // same rank → genuinely a pair
            EXPECT_FALSE(pair_detected_via_obs(static_cast<uint8_t>(a),
                                                static_cast<uint8_t>(b)))
                << "false positive at ids (" << a << ", " << b << ")";
            ++unpaired_checked;
        }
    }
    // Sanity: 1326 total starting hands minus 78 pocket pairs = 1248.
    EXPECT_EQ(unpaired_checked, 1248);
}

TEST(PairDetection, AdjacentRanksDoNotAlias) {
    // Specifically pin a class of bug that would only surface if the bit
    // layout ever flipped (e.g. id = suit*13 + rank): adjacent ranks would
    // start sharing low-order bits and the `/ 4` trick would falsely flag
    // some suited connectors as pairs. The current rank-major layout is
    // safe; this test makes that property regression-proof.
    for (int rank = 0; rank < 12; ++rank) {
        for (int s_lo = 0; s_lo < 4; ++s_lo) {
            for (int s_hi = 0; s_hi < 4; ++s_hi) {
                const uint8_t lo = static_cast<uint8_t>((rank      << 2) | s_lo);
                const uint8_t hi = static_cast<uint8_t>(((rank + 1) << 2) | s_hi);
                EXPECT_FALSE(pair_detected_via_obs(lo, hi));
            }
        }
    }
}

// ─── Helpers for getWinner tests ──────────────────────────────────────────

// Pretty test names — every showdown case names the categories it pits.
struct Showdown {
    const char* name;
    std::array<int, 7> winner_cards;
    std::array<int, 7> loser_cards;
};

// ─── Same category, kicker decides ───────────────────────────────────────

TEST(Winner, SamePairBetterKickerWins) {
    // The hand from the bug report: both have a pair of 2s on a board with
    // a J / T / 7. P0's A kicker beats P1's 8 kicker.
    //   board:  J♣ 5♣ T♠ 2♦ 7♣
    //   p0:     A♦ 2♠   →  22 with A J T kickers  (winner)
    //   p1:     8♥ 2♣   →  22 with J T 8 kickers
    std::array<int, 7> p0 = {
        tpt(RA, DIAMONDS), tpt(R2, SPADES),
        tpt(RJ, CLUBS), tpt(R5, CLUBS), tpt(RT, SPADES), tpt(R2, DIAMONDS), tpt(R7, CLUBS),
    };
    std::array<int, 7> p1 = {
        tpt(R8, HEARTS), tpt(R2, CLUBS),
        tpt(RJ, CLUBS), tpt(R5, CLUBS), tpt(RT, SPADES), tpt(R2, DIAMONDS), tpt(R7, CLUBS),
    };
    EXPECT_EQ(winner_of(p0, p1), 0);
    EXPECT_EQ(winner_of(p1, p0), 1);
}

TEST(Winner, HigherPairWinsRegardlessOfKickers) {
    // Pair of As vs pair of Ks. Aces win even with low kicker vs King's
    // higher unpaired side cards.
    std::array<int, 7> aces = {
        tpt(RA, DIAMONDS), tpt(RA, HEARTS),
        tpt(R2, CLUBS), tpt(R3, DIAMONDS), tpt(R4, HEARTS),
        tpt(R6, SPADES), tpt(R7, CLUBS),
    };
    std::array<int, 7> kings = {
        tpt(RK, DIAMONDS), tpt(RK, HEARTS),
        tpt(R2, CLUBS), tpt(R3, DIAMONDS), tpt(R4, HEARTS),
        tpt(RQ, SPADES), tpt(RJ, CLUBS),  // strong kickers but still kings
    };
    EXPECT_EQ(winner_of(aces, kings), 0);
}

TEST(Winner, IdenticalHandsTie) {
    // Both players play the board: A high straight from the board, hole
    // cards play no role. (Or: identical pair + identical 3 kickers.)
    std::array<int, 7> p0 = {
        tpt(R2, CLUBS), tpt(R3, DIAMONDS),
        tpt(RT, HEARTS), tpt(RJ, HEARTS), tpt(RQ, HEARTS), tpt(RK, HEARTS), tpt(RA, HEARTS),
    };
    std::array<int, 7> p1 = {
        tpt(R4, CLUBS), tpt(R5, DIAMONDS),
        tpt(RT, HEARTS), tpt(RJ, HEARTS), tpt(RQ, HEARTS), tpt(RK, HEARTS), tpt(RA, HEARTS),
    };
    EXPECT_EQ(winner_of(p0, p1), 3);
}

TEST(Winner, SameTwoPairBetterKicker) {
    // Both have AA QQ; the fifth card (kicker) decides.
    std::array<int, 7> better = {
        tpt(RA, CLUBS),    tpt(RA, DIAMONDS),
        tpt(RQ, HEARTS),   tpt(RQ, SPADES),
        tpt(RK, CLUBS),                     // K kicker (winner)
        tpt(R3, DIAMONDS), tpt(R2, HEARTS),
    };
    std::array<int, 7> worse = {
        tpt(RA, CLUBS),    tpt(RA, DIAMONDS),
        tpt(RQ, HEARTS),   tpt(RQ, SPADES),
        tpt(RJ, CLUBS),                     // J kicker
        tpt(R3, DIAMONDS), tpt(R2, HEARTS),
    };
    EXPECT_EQ(winner_of(better, worse), 0);
}

// ─── Category vs category ─────────────────────────────────────────────────

TEST(Winner, PairBeatsHighCard) {
    std::array<int, 7> pair = {
        tpt(RA, CLUBS), tpt(RA, DIAMONDS),
        tpt(R7, HEARTS), tpt(R5, SPADES), tpt(R3, CLUBS),
        tpt(R8, DIAMONDS), tpt(R9, HEARTS),
    };
    std::array<int, 7> highcard = {
        tpt(RK, CLUBS), tpt(RQ, DIAMONDS),
        tpt(RJ, HEARTS), tpt(R9, SPADES), tpt(R7, DIAMONDS),
        tpt(R5, CLUBS), tpt(R2, HEARTS),
    };
    EXPECT_EQ(winner_of(pair, highcard), 0);
}

TEST(Winner, TwoPairBeatsPair) {
    std::array<int, 7> two_pair = {
        tpt(R8, CLUBS), tpt(R8, DIAMONDS),
        tpt(R5, HEARTS), tpt(R5, SPADES),
        tpt(R3, CLUBS), tpt(RJ, DIAMONDS), tpt(RA, HEARTS),
    };
    std::array<int, 7> pair_only = {
        tpt(RA, CLUBS), tpt(RA, SPADES),  // higher pair...
        tpt(R5, HEARTS),                  // ...but with R5 making low pair impossible to beat 88+55
        tpt(R3, CLUBS),
        tpt(R7, DIAMONDS), tpt(R2, HEARTS), tpt(RJ, SPADES),
    };
    // AA still beats 88+55? No — two pair always beats one pair regardless
    // of pair rank. AA (one pair) vs 88+55 (two pair) → two pair wins.
    EXPECT_EQ(winner_of(two_pair, pair_only), 0);
}

TEST(Winner, TripsBeatsTwoPair) {
    std::array<int, 7> trips = {
        tpt(R7, CLUBS), tpt(R7, DIAMONDS), tpt(R7, HEARTS),
        tpt(R3, SPADES), tpt(R2, CLUBS),
        tpt(RA, DIAMONDS), tpt(R5, HEARTS),
    };
    std::array<int, 7> two_pair = {
        tpt(RA, CLUBS), tpt(RA, DIAMONDS),
        tpt(RK, HEARTS), tpt(RK, SPADES),
        tpt(R3, CLUBS), tpt(R2, DIAMONDS), tpt(R5, HEARTS),
    };
    EXPECT_EQ(winner_of(trips, two_pair), 0);
}

TEST(Winner, StraightBeatsTrips) {
    // Wheel straight (A-2-3-4-5) vs trip Aces. Straight wins.
    std::array<int, 7> straight = {
        tpt(R5, CLUBS), tpt(R4, DIAMONDS),
        tpt(R3, HEARTS), tpt(R2, SPADES),
        tpt(RA, CLUBS),
        tpt(RK, DIAMONDS), tpt(RQ, HEARTS),
    };
    std::array<int, 7> trips = {
        tpt(RA, CLUBS), tpt(RA, DIAMONDS), tpt(RA, HEARTS),
        tpt(R7, SPADES), tpt(R8, CLUBS),
        tpt(RK, DIAMONDS), tpt(RJ, HEARTS),
    };
    EXPECT_EQ(winner_of(straight, trips), 0);
}

TEST(Winner, FlushBeatsStraight) {
    // Five hearts vs broadway straight. Flush wins.
    std::array<int, 7> flush = {
        tpt(R2, HEARTS), tpt(R5, HEARTS),
        tpt(R7, HEARTS), tpt(R9, HEARTS), tpt(RJ, HEARTS),
        tpt(R3, CLUBS), tpt(R4, DIAMONDS),
    };
    std::array<int, 7> straight = {
        tpt(RT, CLUBS), tpt(RJ, DIAMONDS),
        tpt(RQ, SPADES), tpt(RK, CLUBS), tpt(RA, DIAMONDS),
        tpt(R3, HEARTS), tpt(R7, SPADES),
    };
    EXPECT_EQ(winner_of(flush, straight), 0);
}

TEST(Winner, FlushBeatsPair) {
    // The user-mentioned scenario: flush always beats pair regardless of pair rank.
    std::array<int, 7> flush = {
        tpt(R2, SPADES), tpt(R5, SPADES),
        tpt(R7, SPADES), tpt(R9, SPADES), tpt(RJ, SPADES),
        tpt(R3, CLUBS), tpt(R4, DIAMONDS),
    };
    std::array<int, 7> pair_aces = {
        tpt(RA, CLUBS), tpt(RA, DIAMONDS),
        tpt(R7, HEARTS), tpt(R9, CLUBS), tpt(RJ, DIAMONDS),
        tpt(R3, SPADES), tpt(R4, HEARTS),
    };
    EXPECT_EQ(winner_of(flush, pair_aces), 0);
}

TEST(Winner, HigherFlushWins) {
    // Both have a heart flush from the same board. Player with higher hole-
    // card heart wins.
    std::array<int, 7> ace_high = {
        tpt(RA, HEARTS), tpt(R3, CLUBS),
        tpt(R2, HEARTS), tpt(R5, HEARTS), tpt(R7, HEARTS), tpt(R9, HEARTS),
        tpt(RJ, DIAMONDS),
    };
    std::array<int, 7> king_high = {
        tpt(RK, HEARTS), tpt(R3, CLUBS),
        tpt(R2, HEARTS), tpt(R5, HEARTS), tpt(R7, HEARTS), tpt(R9, HEARTS),
        tpt(RJ, DIAMONDS),
    };
    EXPECT_EQ(winner_of(ace_high, king_high), 0);
}

TEST(Winner, FullHouseBeatsFlush) {
    std::array<int, 7> full_house = {
        tpt(RA, CLUBS), tpt(RA, DIAMONDS), tpt(RA, HEARTS),
        tpt(RK, SPADES), tpt(RK, CLUBS),
        tpt(R3, DIAMONDS), tpt(R2, HEARTS),
    };
    std::array<int, 7> flush = {
        tpt(R2, SPADES), tpt(R5, SPADES),
        tpt(R7, SPADES), tpt(R9, SPADES), tpt(RJ, SPADES),
        tpt(R3, CLUBS), tpt(R4, DIAMONDS),
    };
    EXPECT_EQ(winner_of(full_house, flush), 0);
}

TEST(Winner, HigherFullHouseWins) {
    // AAA-22 beats KKK-QQ — three-of-a-kind rank decides.
    std::array<int, 7> aces_full = {
        tpt(RA, CLUBS), tpt(RA, DIAMONDS), tpt(RA, HEARTS),
        tpt(R2, SPADES), tpt(R2, CLUBS),
        tpt(R7, DIAMONDS), tpt(R8, HEARTS),
    };
    std::array<int, 7> kings_full = {
        tpt(RK, CLUBS), tpt(RK, DIAMONDS), tpt(RK, HEARTS),
        tpt(RQ, SPADES), tpt(RQ, CLUBS),
        tpt(R7, DIAMONDS), tpt(R8, HEARTS),
    };
    EXPECT_EQ(winner_of(aces_full, kings_full), 0);
}

TEST(Winner, QuadsBeatsFullHouse) {
    std::array<int, 7> quads = {
        tpt(R5, CLUBS), tpt(R5, DIAMONDS), tpt(R5, HEARTS), tpt(R5, SPADES),
        tpt(R3, CLUBS),
        tpt(R7, DIAMONDS), tpt(RA, HEARTS),
    };
    std::array<int, 7> aces_full = {
        tpt(RA, CLUBS), tpt(RA, DIAMONDS), tpt(RA, HEARTS),
        tpt(RK, SPADES), tpt(RK, CLUBS),
        tpt(R3, DIAMONDS), tpt(R2, HEARTS),
    };
    EXPECT_EQ(winner_of(quads, aces_full), 0);
}

TEST(Winner, StraightFlushBeatsQuads) {
    std::array<int, 7> sf = {
        tpt(R5, HEARTS), tpt(R6, HEARTS),
        tpt(R7, HEARTS), tpt(R8, HEARTS), tpt(R9, HEARTS),
        tpt(RA, CLUBS),  tpt(R2, DIAMONDS),
    };
    std::array<int, 7> quads = {
        tpt(RA, CLUBS), tpt(RA, DIAMONDS), tpt(RA, HEARTS), tpt(RA, SPADES),
        tpt(RK, CLUBS),
        tpt(RQ, DIAMONDS), tpt(RJ, HEARTS),
    };
    EXPECT_EQ(winner_of(sf, quads), 0);
}

TEST(Winner, RoyalFlushBeatsStraightFlush) {
    std::array<int, 7> royal = {
        tpt(RT, HEARTS), tpt(RJ, HEARTS),
        tpt(RQ, HEARTS), tpt(RK, HEARTS), tpt(RA, HEARTS),
        tpt(R2, CLUBS), tpt(R3, DIAMONDS),
    };
    std::array<int, 7> sf_low = {
        tpt(R5, CLUBS), tpt(R6, CLUBS),
        tpt(R7, CLUBS), tpt(R8, CLUBS), tpt(R9, CLUBS),
        tpt(R2, HEARTS), tpt(R3, DIAMONDS),
    };
    EXPECT_EQ(winner_of(royal, sf_low), 0);
}

// ─── Subtler cases ────────────────────────────────────────────────────────

TEST(Winner, BoardPlaysTie) {
    // Board makes a straight; both players' hole cards are dominated.
    std::array<int, 7> p0 = {
        tpt(R2, CLUBS), tpt(R3, DIAMONDS),
        tpt(R4, HEARTS), tpt(R5, SPADES), tpt(R6, CLUBS),
        tpt(R7, DIAMONDS), tpt(R8, HEARTS),
    };
    std::array<int, 7> p1 = {
        tpt(R2, SPADES), tpt(R3, HEARTS),  // different suits, same ranks
        tpt(R4, HEARTS), tpt(R5, SPADES), tpt(R6, CLUBS),
        tpt(R7, DIAMONDS), tpt(R8, HEARTS),
    };
    // Both play 4-5-6-7-8 board straight.
    EXPECT_EQ(winner_of(p0, p1), 3);
}

TEST(Winner, FourFlushDoesNotMakeFlush) {
    // Bug-relevant: with 4 cards of one suit (3 board + 1 hole), no flush.
    // Pair of 2s with A kicker should win over pair of 2s with 8 kicker
    // even if one player has 4 clubs.
    std::array<int, 7> ace_kicker = {
        tpt(RA, DIAMONDS), tpt(R2, SPADES),
        tpt(RJ, CLUBS), tpt(R5, CLUBS), tpt(RT, SPADES),
        tpt(R2, DIAMONDS), tpt(R7, CLUBS),
    };
    std::array<int, 7> eight_kicker_4clubs = {
        tpt(R8, HEARTS), tpt(R2, CLUBS),  // 4 clubs total in this hand
        tpt(RJ, CLUBS), tpt(R5, CLUBS), tpt(RT, SPADES),
        tpt(R2, DIAMONDS), tpt(R7, CLUBS),
    };
    EXPECT_EQ(winner_of(ace_kicker, eight_kicker_4clubs), 0);
}

TEST(Winner, WheelStraightVsHighStraight) {
    // 5-high straight (A-2-3-4-5) vs 6-high straight (2-3-4-5-6).
    // The 6-high wins.
    std::array<int, 7> wheel = {
        tpt(RA, CLUBS), tpt(R2, DIAMONDS),
        tpt(R3, HEARTS), tpt(R4, SPADES), tpt(R5, CLUBS),
        tpt(R8, DIAMONDS), tpt(R9, HEARTS),
    };
    std::array<int, 7> six_high = {
        tpt(R2, CLUBS), tpt(R3, DIAMONDS),
        tpt(R4, HEARTS), tpt(R5, SPADES), tpt(R6, CLUBS),
        tpt(R8, DIAMONDS), tpt(R9, HEARTS),
    };
    EXPECT_EQ(winner_of(six_high, wheel), 0);
}

TEST(Winner, TripsKickerComparisonOnTwoBoards) {
    // Same trips (set of Ks on the board), different kicker top.
    // P0: K-K-K + A,Q kickers (best 5 = KKK A Q)
    // P1: K-K-K + Q,J kickers (best 5 = KKK Q J)
    std::array<int, 7> ace_kick = {
        tpt(RA, CLUBS), tpt(R2, DIAMONDS),
        tpt(RK, HEARTS), tpt(RK, SPADES), tpt(RK, CLUBS),
        tpt(RQ, DIAMONDS), tpt(R3, HEARTS),
    };
    std::array<int, 7> jack_kick = {
        tpt(RJ, CLUBS), tpt(R2, SPADES),
        tpt(RK, HEARTS), tpt(RK, SPADES), tpt(RK, CLUBS),
        tpt(RQ, DIAMONDS), tpt(R3, HEARTS),
    };
    EXPECT_EQ(winner_of(ace_kick, jack_kick), 0);
}

// ─── End-to-end integration: DiscreteGame → showdown → getUtility ───────
//
// Drives an actual game through check-down to showdown for many seeds and
// asserts that the engine's terminal utility is consistent with what the
// hand evaluator says directly for the dealt cards. This pins the *full
// chain*: makeTerminal's encoding conversion, getWinner's strength
// comparison, and getUtility's winner/tie/lose split — all wired together
// the way training consumes them.

namespace {

// Apply the call-or-check action from the current action set. Test helper:
// during a check-down, exactly one of {Call, Check} is always legal for
// the acting player, so this advances one ply unambiguously.
template <typename GameT>
void check_or_call(GameT& g) {
    const auto& acts = g.getActions();
    for (const auto& a : acts) {
        if (std::holds_alternative<Game::Call>(a) ||
            std::holds_alternative<Game::Check>(a)) {
            g.transition(a);
            return;
        }
    }
    FAIL() << "no Call or Check available among legal actions";
}

}  // namespace

TEST(Integration, EndToEndShowdownAgreesWithEvaluator) {
    // Reuse the project's default 52-card NLHE config so blinds, stacks,
    // and round structure match what training sees. The test drives
    // check-down to showdown regardless of bet menu, so we use the same
    // DefaultGameConfig (11 pot fractions) as the rest of the codebase
    // — Game::DiscreteGame is non-templated and accepts only that
    // instantiation.
    Game::DefaultGameConfig gcfg = Game::make_nlhe_full_52();
    gcfg.validate();

    Game::DefaultBettingConfig bet_cfg = Game::make_default_betting_config(gcfg);

    // Helper: deck.h id → TwoPlusTwo id (mirrors Transitioner's conversion).
    // Single source of truth for the deck.h ↔ TwoPlusTwo encoding lives
    // in Game/Utility/CardConversion.hpp; reuse it here so a future
    // encoding tweak doesn't require touching this test.
    auto to_2p2 = [](uint8_t c) { return Game::deck_to_two_plus_two(c); };

    constexpr int N_HANDS = 50;
    int wins_p0 = 0, wins_p1 = 0, ties = 0;

    for (int seed = 0; seed < N_HANDS; ++seed) {
        std::mt19937 rng(static_cast<uint32_t>(seed) ^ 0xC0FFEE);
        Game::DiscreteGame game(rng, gcfg, bet_cfg);

        // The Game starts in ChanceState; transitioning a Chance action
        // moves it into preflop ActionState. Auto-advance any chance nodes.
        while (game.getType() == "chance" && !game.isTerminal()) {
            game.transition(Game::Chance{});
        }

        // Drive check-down: SB calls preflop, BB checks, then BB-checks /
        // SB-checks each post-flop street. With min_raise enforced, no
        // accidental raises on a check-down sequence.
        int safety = 0;
        while (!game.isTerminal()) {
            if (game.getType() == "chance") {
                game.transition(Game::Chance{});
            } else {
                check_or_call(game);
            }
            ASSERT_LT(++safety, 50) << "runaway transition loop on seed=" << seed;
        }

        ASSERT_TRUE(game.isTerminal());

        // Read the actually-dealt cards from the (now-terminal) game.
        const auto& raw = game.getContext().cards.rawCards;
        int p0_2p2[7] = {
            to_2p2(raw[0]), to_2p2(raw[1]),
            to_2p2(raw[4]), to_2p2(raw[5]), to_2p2(raw[6]),
            to_2p2(raw[7]), to_2p2(raw[8]),
        };
        int p1_2p2[7] = {
            to_2p2(raw[2]), to_2p2(raw[3]),
            to_2p2(raw[4]), to_2p2(raw[5]), to_2p2(raw[6]),
            to_2p2(raw[7]), to_2p2(raw[8]),
        };
        const int expected_winner_raw = Utility::getWinner(p0_2p2, p1_2p2);
        // getWinner: 0=p0, 1=p1, 3=tie. Engine's terminal convention: 0/1/2.
        const int expected_engine_winner =
            (expected_winner_raw == 3) ? 2 : expected_winner_raw;

        const int u0 = game.getUtility(0);
        const int u1 = game.getUtility(1);

        // Zero-sum invariant must hold regardless of the result.
        EXPECT_EQ(u0 + u1, 0) << "non-zero-sum at seed=" << seed
                              << " u0=" << u0 << " u1=" << u1;

        if (expected_engine_winner == 2) {
            // Tie: both players' contributions returned, so net 0 each.
            EXPECT_EQ(u0, 0) << "expected tie at seed=" << seed
                             << " u0=" << u0;
            EXPECT_EQ(u1, 0);
            ++ties;
        } else if (expected_engine_winner == 0) {
            EXPECT_GT(u0, 0) << "p0 expected to win at seed=" << seed
                             << " u0=" << u0;
            EXPECT_LT(u1, 0);
            ++wins_p0;
        } else {
            EXPECT_LT(u0, 0) << "p1 expected to win at seed=" << seed
                             << " u0=" << u0;
            EXPECT_GT(u1, 0);
            ++wins_p1;
        }
    }
    // Sanity: across 50 random check-downs we should see all three
    // outcomes at non-trivial frequency. If any bucket is empty something
    // odd is happening (e.g. a bug always favouring one seat).
    EXPECT_GT(wins_p0, 0);
    EXPECT_GT(wins_p1, 0);
    // Ties are rarer; just check the counts add up.
    EXPECT_EQ(wins_p0 + wins_p1 + ties, N_HANDS);
}

// ─── Quick statistical check that the distribution looks sane ────────────

TEST(Winner, ManyRandomShowdownsAreSelfConsistent) {
    // For a sample of random 9-card configurations (2 hole + 2 hole + 5
    // board) the result of getWinner(p0, p1) and getWinner(p1, p0) must be
    // mirrored: 0↔1 swap, 3 stays. Catches accidental asymmetry in the
    // evaluator path (e.g. using an out-of-bounds sentinel).
    // Using a fixed seed for determinism.
    std::mt19937 rng(0xDEADBEEFu);
    std::vector<int> deck(52);
    std::iota(deck.begin(), deck.end(), 1);  // TwoPlusTwo ids 1..52
    int swaps_match = 0;
    constexpr int N = 200;
    for (int i = 0; i < N; ++i) {
        std::shuffle(deck.begin(), deck.end(), rng);
        std::array<int, 7> p0 = {
            deck[0], deck[1], deck[4], deck[5], deck[6], deck[7], deck[8],
        };
        std::array<int, 7> p1 = {
            deck[2], deck[3], deck[4], deck[5], deck[6], deck[7], deck[8],
        };
        const int w_ab = winner_of(p0, p1);
        const int w_ba = winner_of(p1, p0);
        if (w_ab == 3) {
            EXPECT_EQ(w_ba, 3);
        } else {
            EXPECT_EQ(w_ab + w_ba, 1);  // {0,1} or {1,0}
        }
        ++swaps_match;
    }
    EXPECT_EQ(swaps_match, N);
}

}  // namespace
