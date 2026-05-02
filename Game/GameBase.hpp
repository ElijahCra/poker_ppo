//
// Created by Elijah Crain on 10/5/25.
//

#ifndef CFR2_TEXAS_GAMEBASE_HPP
#define CFR2_TEXAS_GAMEBASE_HPP
#include <optional>
#include <string>
#include <variant>

#include "GameConfig.hpp"

namespace Game {

// Structural constants — fixed at build time because arrays, blind layout,
// and the static hand_indexer assume them. Runtime variation lives in
// DefaultGameConfig (initial_stack, blinds, max_raises, excluded_cards, …).
static constexpr int NUM_PLAYERS         = NUM_PLAYERS_FIXED;
static constexpr int NUM_ROUNDS          = NUM_ROUNDS_FIXED;
static constexpr int DECK_SIZE           = CARD_NAMESPACE_SIZE;  // card-ID space (always 52)
static constexpr int NUM_HOLE_CARDS      = NUM_HOLE_CARDS_FIXED;
static constexpr int NUM_COMMUNITY_CARDS = 5;

// Action types as structs
struct Fold {
    static constexpr auto name = "fold";
};

struct Check {
    static constexpr auto name = "check";
};

struct Call {
    static constexpr auto name = "call";
    uint32_t amount = 0; // Amount to call / added to pot by calling in mbb (millibigblinds)

    explicit Call(const uint32_t amt) : amount(amt) {}
};

struct Raise {
    static constexpr auto name = "raise";
    uint32_t amount = 0;  // Amount added to pot in mbb (millibigblinds)

    explicit Raise(const uint32_t amt) : amount(amt) {}
};

struct Chance { // action for chance player (i.e. dealing cards)
    static constexpr auto name = "chance";
};

// Variant holding all possible actions
using Action = std::variant<
    Fold,
    Check,
    Call,
    Raise,
    Chance
>;
}

#endif //CFR2_TEXAS_GAMEBASE_HPP