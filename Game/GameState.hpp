#ifndef CFR2_TEXAS_GAME_STATE_HPP
#define CFR2_TEXAS_GAME_STATE_HPP
#include <variant>

namespace Game {
// State data structures - each holds its own state-specific data
struct ChanceState {
    static constexpr auto name = "chance";
    bool cardsDealt = false;
    enum Type { DEAL_HOLE, DEAL_FLOP, DEAL_TURN, DEAL_RIVER } type = DEAL_HOLE;
};

struct ActionState {
    static constexpr auto name = "action";
    bool firstActionTaken = false;
    uint8_t checksCount = 0;
    uint32_t currentBet = 0;
    uint8_t raiseCount = 0;
    uint8_t lastRaiser = -1;
};

struct TerminalState {
    static constexpr auto name = "terminal";
    uint8_t winner = -1;
    double pot = 0;
    enum Reason { SHOWDOWN, FOLD, ALL_IN } reason = SHOWDOWN;
};

// Variant holding all possible states
using GameState = std::variant<ChanceState, ActionState, TerminalState>;
}
#endif //CFR2_TEXAS_GAME_STATE_HPP
