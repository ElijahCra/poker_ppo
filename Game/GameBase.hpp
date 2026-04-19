//
// Created by Elijah Crain on 10/5/25.
//

#ifndef CFR2_TEXAS_GAMEBASE_HPP
#define CFR2_TEXAS_GAMEBASE_HPP
#include <optional>
#include <string>
#include <variant>

namespace Game {

// Game constants
static constexpr int NUM_PLAYERS = 2;
static constexpr int NUM_ROUNDS = 4;
static constexpr int DECK_SIZE = 52;
static constexpr int NUM_HOLE_CARDS = 2;
static constexpr int NUM_COMMUNITY_CARDS = 5;
static constexpr int MAX_RAISES = 4;
static constexpr uint32_t INITIAL_STACK = 100'000; // 100bb = 100,000 mbb

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