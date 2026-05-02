//
// Compile-time game configuration. Templated on the bet-sizing menu length
// and the count of excluded cards so a concrete instance can be a
// constexpr value, suitable for `inline constexpr` globals and `if constexpr`
// gating on its fields.
//
// Pattern: ./PPO_CFR/TexasGame/Utils.hpp — std::array sized by template
// parameters, std::string_view for names, default-initialised POD fields.
//
// Card encoding convention (unchanged):
//   card_id = rank * 4 + suit,   rank ∈ [0..12] (0 = 2, 12 = A), suit ∈ [0..3].
// A "reduced deck" is expressed via the `excluded_cards` std::array; the dealt
// cards remain in the standard 0..51 namespace so the TwoPlusTwo hand
// evaluator and hand_indexer continue to work without modification.
//

#ifndef POKER_PPO_GAME_GAMECONFIG_HPP
#define POKER_PPO_GAME_GAMECONFIG_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace Game {

// Fixed structural constants. Widening any of these requires deeper
// surgery (hand_indexer init, blind logic, evaluator). Documented and
// asserted against in `validate()`.
inline constexpr int    CARD_NAMESPACE_SIZE  = 52;
inline constexpr int    NUM_PLAYERS_FIXED    = 2;
inline constexpr int    NUM_ROUNDS_FIXED     = 4;
inline constexpr int    NUM_HOLE_CARDS_FIXED = 2;
inline constexpr std::array<uint8_t, NUM_ROUNDS_FIXED>
    CARDS_PER_ROUND_FIXED = {2, 3, 1, 1};

// Templated on the **count** of pot-fraction raise sizes and **count** of
// excluded cards. Default = 3 fractions (matching the original
// {0.5, 1.0, 2.0} menu) and zero excluded cards (full 52-card deck).
template <std::size_t NumPotFractions  = 3,
          std::size_t NumExcludedCards = 0>
struct GameConfig {
    // Compile-time exposure so callers can read the template parameters.
    static constexpr std::size_t kNumPotFractions  = NumPotFractions;
    static constexpr std::size_t kNumExcludedCards = NumExcludedCards;

    // ─── Players (currently fixed at 2; field present for future) ──────
    uint8_t num_players    = NUM_PLAYERS_FIXED;
    uint8_t num_hole_cards = NUM_HOLE_CARDS_FIXED;

    // ─── Rounds (currently fixed; cards_per_round kept for API parity) ─
    uint8_t num_rounds = NUM_ROUNDS_FIXED;
    std::array<uint8_t, NUM_ROUNDS_FIXED> cards_per_round = CARDS_PER_ROUND_FIXED;

    // ─── Cards ─────────────────────────────────────────────────────────
    // Card IDs (0..51) excluded from the dealable deck.
    std::array<uint8_t, NumExcludedCards> excluded_cards{};

    // ─── Money (mbb — 1000 mbb = 1 big blind) ──────────────────────────
    uint32_t initial_stack = 100'000;
    uint32_t small_blind   = 500;
    uint32_t big_blind     = 1000;
    uint32_t min_bet       = 1000;
    uint32_t min_raise     = 1000;

    // ─── Betting structure ─────────────────────────────────────────────
    uint8_t                            max_raises_per_round = 6;
    std::array<double, NumPotFractions> pot_fractions{};
    bool                               include_all_in_slot  = true;

    // ─── Meta ──────────────────────────────────────────────────────────
    // string_view is constexpr-constructible; pass by value, compare with ==.
    std::string_view name = "nlhe_full";
    uint64_t         seed = 0x12345ULL;

    // ─── Derived (constexpr where possible) ────────────────────────────
    [[nodiscard]] constexpr uint8_t deck_size() const noexcept {
        return static_cast<uint8_t>(CARD_NAMESPACE_SIZE - NumExcludedCards);
    }

    [[nodiscard]] constexpr uint8_t total_community_cards() const noexcept {
        uint8_t n = 0;
        for (uint8_t i = 1; i < num_rounds; ++i) {
            n = static_cast<uint8_t>(n + cards_per_round[i]);
        }
        return n;
    }

    [[nodiscard]] constexpr uint8_t total_game_cards() const noexcept {
        return static_cast<uint8_t>(num_players * num_hole_cards
                                  + total_community_cards());
    }

    [[nodiscard]] constexpr int num_raise_slots() const noexcept {
        return static_cast<int>(NumPotFractions)
             + (include_all_in_slot ? 1 : 0);
    }

    [[nodiscard]] constexpr int action_count() const noexcept {
        return 2 + num_raise_slots();
    }

    // ─── Runtime helpers ───────────────────────────────────────────────
    // Returns ordered list of card IDs still in play. Allocates — runtime
    // path (called from CardData::initialize during shuffle setup), so not
    // constexpr.
    [[nodiscard]] std::vector<uint8_t> dealable_cards() const {
        std::vector<uint8_t> cards;
        cards.reserve(deck_size());
        for (uint8_t c = 0; c < CARD_NAMESPACE_SIZE; ++c) {
            bool excluded = false;
            for (uint8_t e : excluded_cards) {
                if (e == c) { excluded = true; break; }
            }
            if (!excluded) cards.push_back(c);
        }
        return cards;
    }

    // Validate against the structural constants this build supports. Used
    // by main.cpp at startup; kept non-constexpr (throws on failure) so
    // diagnostics include a runtime message.
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
};

// ─── Concrete typedefs + constexpr factory functions ──────────────────
//
// To keep the template parameters from going viral through CardData /
// GameContext / DiscreteGame / PokerEnvironment, downstream code uses
// `Game::DefaultGameConfig` (a single concrete instantiation matching
// the live training config in include/config.h). Switching variants is
// done by re-pointing this alias and rebuilding.

inline constexpr std::size_t DEFAULT_NUM_POT_FRACTIONS  = 11;  // 0.25..3.0
inline constexpr std::size_t DEFAULT_NUM_EXCLUDED_CARDS = 0;   // full 52-card

using DefaultGameConfig =
    GameConfig<DEFAULT_NUM_POT_FRACTIONS, DEFAULT_NUM_EXCLUDED_CARDS>;

// Full 52-card NLHE with the 11-fraction bet menu used by main.cpp /
// config.h. Constexpr-constructible for use as an `inline constexpr`
// global.
[[nodiscard]] constexpr DefaultGameConfig make_nlhe_full_52() noexcept {
    DefaultGameConfig cfg;
    cfg.name                 = "nlhe_full_52";
    cfg.initial_stack        = 100'000;
    cfg.small_blind          = 500;
    cfg.big_blind            = 1000;
    cfg.min_bet              = 1000;
    cfg.min_raise            = 1000;
    cfg.max_raises_per_round = 4;
    cfg.pot_fractions        = {
        0.25, 0.33, 0.5, 0.66, 0.75,
        1.0,  1.25, 1.5, 2.0,  2.5, 3.0,
    };
    cfg.include_all_in_slot  = true;
    return cfg;
}

// ─── Live constexpr instance ──────────────────────────────────────────
//
// `kGameConfig` is the single compile-time DefaultGameConfig consumed by
// the runtime. Downstream code references `Game::kGameConfig` instead of
// re-invoking `make_nlhe_full_52()` per call site, so:
//   - the bytes live in `.rodata` rather than a fresh stack frame
//   - `static_assert`s on its fields work at namespace scope
//   - `Game.hpp` / `poker_env.h` defaults bind to the same exact value
// To switch variants, swap the factory call here and rebuild.
inline constexpr DefaultGameConfig kGameConfig = make_nlhe_full_52();

// Pack ranks-to-remove (0..12) into the card-ID exclusion list at
// compile time. Returns std::array sized by `NumRanksToRemove * 4`.
template <std::size_t NumRanksToRemove>
[[nodiscard]] constexpr std::array<uint8_t, NumRanksToRemove * 4>
cardsOfRanksArr(const std::array<uint8_t, NumRanksToRemove>& ranks) noexcept {
    std::array<uint8_t, NumRanksToRemove * 4> cards{};
    std::size_t out = 0;
    for (uint8_t r : ranks) {
        for (uint8_t s = 0; s < 4; ++s) cards[out++] = static_cast<uint8_t>(r * 4 + s);
    }
    return cards;
}

}  // namespace Game

#endif  // POKER_PPO_GAME_GAMECONFIG_HPP
