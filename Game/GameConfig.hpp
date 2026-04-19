//
// Runtime game configuration. One GameConfig describes a full poker variant:
// card space, stacks, blinds, betting structure. Pass by value/const ref into
// Game/GameContext so a single binary can instantiate many different variants
// (useful for sweep harnesses).
//
// Card encoding convention (unchanged from the rest of the engine):
//   card_id = rank * 4 + suit,   rank ∈ [0..12] (0 = 2, 12 = A), suit ∈ [0..3].
// A "reduced deck" is expressed by listing card_ids to exclude. The dealt cards
// remain in the standard 0..51 namespace so the TwoPlusTwo hand evaluator and
// hand_indexer continue to work without modification.
//

#ifndef POKER_PPO_GAME_GAMECONFIG_HPP
#define POKER_PPO_GAME_GAMECONFIG_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace Game {

// Fixed structural constants. Widening any of these requires deeper surgery
// (array sizes, hand_indexer init, blind logic). Document & assert against.
inline constexpr int    CARD_NAMESPACE_SIZE = 52;  // card_id range [0..51]
inline constexpr int    NUM_PLAYERS_FIXED   = 2;
inline constexpr int    NUM_ROUNDS_FIXED    = 4;   // preflop, flop, turn, river
inline constexpr int    NUM_HOLE_CARDS_FIXED = 2;
inline constexpr std::array<uint8_t, NUM_ROUNDS_FIXED>
    CARDS_PER_ROUND_FIXED = {2, 3, 1, 1};          // indexer format

struct GameConfig {
    // ─── Players (currently fixed at 2; field present for future) ──────
    uint8_t num_players   = NUM_PLAYERS_FIXED;
    uint8_t num_hole_cards = NUM_HOLE_CARDS_FIXED;

    // ─── Rounds (currently fixed; cards_per_round kept for API parity) ─
    uint8_t num_rounds = NUM_ROUNDS_FIXED;
    // Hole+board deal schedule. Must match CARDS_PER_ROUND_FIXED for now
    // because the static hand_indexer is initialized once at that shape.
    std::array<uint8_t, NUM_ROUNDS_FIXED> cards_per_round = CARDS_PER_ROUND_FIXED;

    // ─── Cards ──────────────────────────────────────────────────────────
    // Card IDs (0..51) to exclude from the dealable deck.
    // Common pattern: remove whole ranks via cardsOfRanks(...).
    std::vector<uint8_t> excluded_cards {};

    // ─── Money (mbb — 1000 mbb = 1 big blind) ───────────────────────────
    uint32_t initial_stack = 100'000;   // 100 bb
    uint32_t small_blind   = 500;
    uint32_t big_blind     = 1000;
    uint32_t min_bet       = 1000;
    uint32_t min_raise     = 1000;

    // ─── Betting structure ──────────────────────────────────────────────
    uint8_t              max_raises_per_round = 4;
    std::vector<double>  pot_fractions        = {0.5, 1.0, 2.0};
    bool                 include_all_in_slot  = true;

    // ─── Meta ───────────────────────────────────────────────────────────
    std::string name = "nlhe_full";   // human-readable tag for logging
    uint64_t    seed = 0x12345ULL;

    // ─── Derived ────────────────────────────────────────────────────────
    [[nodiscard]] uint8_t deck_size() const noexcept {
        return static_cast<uint8_t>(CARD_NAMESPACE_SIZE - excluded_cards.size());
    }

    [[nodiscard]] uint8_t total_community_cards() const noexcept {
        uint8_t n = 0;
        for (uint8_t i = 1; i < num_rounds; ++i) n = static_cast<uint8_t>(n + cards_per_round[i]);
        return n;
    }

    [[nodiscard]] uint8_t total_game_cards() const noexcept {
        return static_cast<uint8_t>(num_players * num_hole_cards + total_community_cards());
    }

    [[nodiscard]] int num_raise_slots() const noexcept {
        return static_cast<int>(pot_fractions.size()) + (include_all_in_slot ? 1 : 0);
    }

    [[nodiscard]] int action_count() const noexcept { return 2 + num_raise_slots(); }

    // Ordered list of card IDs still in play.
    [[nodiscard]] std::vector<uint8_t> dealable_cards() const {
        std::vector<uint8_t> cards;
        cards.reserve(deck_size());
        for (uint8_t c = 0; c < CARD_NAMESPACE_SIZE; ++c) {
            if (std::find(excluded_cards.begin(), excluded_cards.end(), c)
                == excluded_cards.end()) {
                cards.push_back(c);
            }
        }
        return cards;
    }

    // Validate against the structural constants this build supports.
    void validate() const {
        if (num_players != NUM_PLAYERS_FIXED)
            throw std::invalid_argument("GameConfig: only num_players=2 is supported");
        if (num_hole_cards != NUM_HOLE_CARDS_FIXED)
            throw std::invalid_argument("GameConfig: only num_hole_cards=2 is supported");
        if (num_rounds != NUM_ROUNDS_FIXED)
            throw std::invalid_argument("GameConfig: only num_rounds=4 is supported");
        if (cards_per_round != CARDS_PER_ROUND_FIXED)
            throw std::invalid_argument("GameConfig: cards_per_round must be {2,3,1,1}");
        if (deck_size() < total_game_cards())
            throw std::invalid_argument("GameConfig: deck too small to deal game cards");
        if (big_blind > initial_stack || small_blind > initial_stack)
            throw std::invalid_argument("GameConfig: blinds exceed initial_stack");
    }

    // ─── Helpers for writing configs ────────────────────────────────────
    // Pack ranks-to-remove (0..12) into the card-ID exclusion list.
    static std::vector<uint8_t> cardsOfRanks(const std::vector<uint8_t>& ranks) {
        std::vector<uint8_t> cards;
        cards.reserve(ranks.size() * 4);
        for (auto r : ranks) {
            for (uint8_t s = 0; s < 4; ++s) cards.push_back(static_cast<uint8_t>(r * 4 + s));
        }
        return cards;
    }

    // Short-deck Hold'em: remove 2s, 3s, 4s, 5s (ranks 0..3) → 36 cards.
    static GameConfig shortDeck36() {
        GameConfig cfg;
        cfg.name = "short_deck_36";
        cfg.excluded_cards = cardsOfRanks({0, 1, 2, 3});
        return cfg;
    }

    // Drop the 2s only → 48-card deck (mild reduction, sanity check).
    static GameConfig noDeuces() {
        GameConfig cfg;
        cfg.name = "no_deuces_48";
        cfg.excluded_cards = cardsOfRanks({0});
        return cfg;
    }
};

}  // namespace Game

#endif  // POKER_PPO_GAME_GAMECONFIG_HPP
