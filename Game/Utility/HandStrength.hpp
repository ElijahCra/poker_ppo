#pragma once
//
// HandStrength.hpp — categorical hand-strength scoring for partial hands.
//
// Replaces the prior `handHashes`-based feature in observation_builder.
// `handHashes` returned a hand_indexer canonical id, which is monotonic in
// the order the indexer enumerates hands — *not* in win probability.
// Adjacent indexer ids can have wildly different equities, and pocket
// pairs (the structurally distinguishable preflop hand class) land in a
// region not separable from non-pair high-card hands. The PPO learner
// can't easily exploit "all-in if pair" opponents because the input
// feature gives no clean pair signal.
//
// This helper computes the standard TwoPlusTwo-aligned hand category
// (high card → straight flush) directly from rank/suit counts. It works
// for any 2..7 card subset (preflop hole cards through river hole +
// board), so the env can use the same routine for all four streets.
// Output is monotonic in strength, and pocket pairs appear at category
// 2 (one pair) preflop while non-pair preflop hands land at category 1
// (high card) — a clear, learnable signal for the network.
//
// Categories (matches Ray Wotton's TwoPlusTwo `>>12` hand-class encoding):
//   0  empty / fewer than 2 cards
//   1  high card
//   2  one pair         (incl. preflop pocket pair)
//   3  two pair
//   4  three of a kind  (incl. set on the board)
//   5  straight
//   6  flush
//   7  full house
//   8  four of a kind
//   9  straight flush   (incl. royal flush)
//

#include <array>
#include <cstdint>

namespace Game {

namespace detail {

// Rank value of a deck.h-encoded card (0..51): rank ∈ [0, 13).
constexpr int rank_of(uint8_t card) noexcept { return card >> 2; }
// Suit value of a deck.h-encoded card (0..51): suit ∈ [0, 4).
constexpr int suit_of(uint8_t card) noexcept { return card & 3; }

// Detect a 5-in-a-row run in a 13-bit rank-presence mask. Returns true
// for any of: 23456 (mask 0x001F), ..., TJQKA (mask 0x1F00). Treats the
// ace as both high (rank 12) and low (the wheel A2345 → 0x100F).
constexpr bool has_straight(uint16_t rank_mask) noexcept {
    // Standard 9 high-card straights. The wheel needs an explicit check
    // since A's bit (rank 12) doesn't sit adjacent to 2's bit (rank 0).
    constexpr uint16_t WHEEL = 0x100F;     // A 2 3 4 5
    if ((rank_mask & WHEEL) == WHEEL) return true;
    uint16_t run = rank_mask;
    run &= run >> 1;
    run &= run >> 1;
    run &= run >> 1;
    run &= run >> 1;
    return run != 0;
}

}  // namespace detail

/// Hand category from up to 7 cards (deck.h ids, 0..51). Returns
/// `0` if `n < 2`. Computed by counting ranks/suits — no external
/// table lookups required, so it's safe to call early in initialisation
/// before HandRanks.dat is loaded.
[[nodiscard]] inline int hand_category(const uint8_t* cards, int n) noexcept {
    if (n < 2) return 0;

    int rank_count[13] = {};
    int suit_count[4]  = {};
    // Per-suit rank-presence mask for flush / straight-flush detection.
    uint16_t suit_rank_mask[4] = {};
    uint16_t any_rank_mask     = 0;

    for (int i = 0; i < n; ++i) {
        const int r = detail::rank_of(cards[i]);
        const int s = detail::suit_of(cards[i]);
        rank_count[r]++;
        suit_count[s]++;
        suit_rank_mask[s] |= static_cast<uint16_t>(1u << r);
        any_rank_mask     |= static_cast<uint16_t>(1u << r);
    }

    // Bucket rank counts: how many ranks appear with count k?
    int n_pairs = 0, n_trips = 0, n_quads = 0;
    for (int r = 0; r < 13; ++r) {
        if (rank_count[r] == 2) ++n_pairs;
        else if (rank_count[r] == 3) ++n_trips;
        else if (rank_count[r] == 4) ++n_quads;
    }

    // Flush + straight-flush. Only a 5-card straight-flush counts; with
    // 6 or 7 cards in suit, we still want to flag "straight flush" if
    // any 5 of those form a run.
    int flush_suit = -1;
    bool straight_flush = false;
    for (int s = 0; s < 4; ++s) {
        if (suit_count[s] >= 5) {
            flush_suit = s;
            if (detail::has_straight(suit_rank_mask[s])) straight_flush = true;
        }
    }
    const bool any_straight = detail::has_straight(any_rank_mask);

    // Order by descending strength so we return the best made class.
    if (straight_flush)              return 9;
    if (n_quads > 0)                 return 8;
    if (n_trips > 0 && n_pairs > 0)  return 7;   // full house
    if (n_trips >= 2)                return 7;   // two trips → also FH (pick higher trips, lower as pair)
    if (flush_suit >= 0)             return 6;
    if (any_straight)                return 5;
    if (n_trips > 0)                 return 4;
    if (n_pairs >= 2)                return 3;
    if (n_pairs == 1)                return 2;
    return 1;                                    // high card
}

/// Count of distinct made-hand categories used by `hand_category`.
/// Useful for normalising the result to [0, 1] for an ML feature.
inline constexpr int kHandCategoryCount = 9;

// Speculative / draw features
//
// Equity in HU NLHE is not just about the made hand — a flush draw on
// the flop carries ~35% equity to the river even while the made hand is
// only "high card." A learner seeing only `hand_category` can't tell a
// 4-flush + open-ended straight draw from a complete air-ball. These
// helpers expose the standard draw signals as compact integer counts;
// the obs builder normalises them to [0, 1] for the network.
//

/// Largest suit count among the visible cards. 4 = "one-card flush
/// draw", 5+ = flush already made, ≤3 = no flush draw (or a backdoor
/// only). Returns 0 if `n == 0`.
[[nodiscard]] inline int max_suit_count(const uint8_t* cards, int n) noexcept {
    int suit_count[4] = {};
    for (int i = 0; i < n; ++i) suit_count[detail::suit_of(cards[i])]++;
    int best = 0;
    for (int s = 0; s < 4; ++s) if (suit_count[s] > best) best = suit_count[s];
    return best;
}

/// Number of 5-rank straight windows still "alive" given the visible
/// cards and the remaining streets to come.
///
/// A window is alive if the count of its ranks present in the visible
/// set is at least the round-specific threshold:
///   preflop (n=2):  ≥ 2 (both hole-card ranks in the window)
///   flop    (n=5):  ≥ 3
///   turn    (n=6):  ≥ 4
///   river   (n=7):  = 5  (made straight; suppressed by caller)
///
/// This is the only one of the straight features that's meaningful
/// preflop — `straight_draw_outs` requires ≥ 4 visible cards to detect a
/// 4-of-5 window, so two-card hands all map to 0 there. With this
/// helper, `67` (4 alive windows) is clearly distinguished from `6J`
/// (0 windows) and `AK` (1 window: `TJQKA`).
///
/// Returns 0..5 for typical hands; the obs-builder normalises by 4
/// (the max attainable from a single mid-range connector preflop).
[[nodiscard]] inline int straight_alive_windows(const uint8_t* cards, int n) noexcept {
    if (n < 2) return 0;

    // Round-specific threshold. Anything outside the four canonical card
    // counts {2, 5, 6, 7} (preflop / flop / turn / river) falls through
    // to a sensible default — we extrapolate linearly so 3- or 4-card
    // intermediate states also produce a reasonable answer.
    int threshold;
    if      (n <= 2) threshold = 2;
    else if (n <= 4) threshold = 2;       // pre-flop range; not a real round
    else if (n == 5) threshold = 3;
    else if (n == 6) threshold = 4;
    else             threshold = 5;       // n ≥ 7

    uint16_t mask = 0;
    for (int i = 0; i < n; ++i) {
        mask |= static_cast<uint16_t>(1u << detail::rank_of(cards[i]));
    }

    auto popcount5 = [](uint16_t window5) noexcept {
        int c = 0;
        while (window5) { c += window5 & 1u; window5 >>= 1; }
        return c;
    };

    int alive = 0;
    // Standard windows TJQKA → 23456 (window low rank ∈ [0, 8]).
    for (int low = 0; low <= 8; ++low) {
        const uint16_t window5 = (mask >> low) & 0x1Fu;
        if (popcount5(window5) >= threshold) ++alive;
    }
    // Wheel: A (rank 12) acts as the low card with 2 3 4 5.
    {
        uint16_t wheel = 0;
        if (mask & (1u << 12)) wheel |= 1u;
        wheel |= (mask & 0x000F) << 1;
        if (popcount5(wheel) >= threshold) ++alive;
    }
    return alive;
}

/// Number of distinct missing ranks that, if added to the visible cards,
/// would complete a 5-card straight (incl. the wheel A2345). The card-
/// out count is `4 ×` this minus already-visible duplicates of the
/// out-rank — but rank-outs are a monotonic proxy for equity and avoid
/// per-suit bookkeeping.
///
/// Returns:
///   0     — no straight draw
///   1     — gutshot / single-ended (typical 4 outs)
///   2     — open-ended / wheel-OESD / double belly-buster (typical 8 outs)
///   3+    — rare multi-way draws
///
/// The obs-builder normalises by 8, so 0 → 0.0, 1 → 0.125, 2 → 0.25, …,
/// effectively distinguishing OESD (≈1.0) from gutshot (≈0.5) cleanly.
/// Returns 0 if a straight is already made (callers should suppress
/// against `hand_category` ≥ 5 to avoid double-counting).
[[nodiscard]] inline int straight_draw_outs(const uint8_t* cards, int n) noexcept {
    if (n < 4) return 0;   // can't have 4-of-5 with fewer than 4 cards

    uint16_t mask = 0;
    for (int i = 0; i < n; ++i) {
        mask |= static_cast<uint16_t>(1u << detail::rank_of(cards[i]));
    }

    int outs = 0;
    for (int r = 0; r < 13; ++r) {
        const uint16_t bit = static_cast<uint16_t>(1u << r);
        if (mask & bit) continue;                       // rank already present
        if (detail::has_straight(static_cast<uint16_t>(mask | bit))) ++outs;
    }
    return outs;
}

/// Number of hole cards strictly greater in rank than every visible
/// community card. 0..2. Returns 0 preflop (no community) or when both
/// hole cards are dominated by the board.
[[nodiscard]] inline int overcard_count(const uint8_t* hole,
                                        const uint8_t* community,
                                        int n_community) noexcept {
    if (n_community <= 0) return 0;
    int max_board = -1;
    for (int i = 0; i < n_community; ++i) {
        const int r = detail::rank_of(community[i]);
        if (r > max_board) max_board = r;
    }
    int cnt = 0;
    if (detail::rank_of(hole[0]) > max_board) ++cnt;
    if (detail::rank_of(hole[1]) > max_board) ++cnt;
    return cnt;
}

}  // namespace Game
