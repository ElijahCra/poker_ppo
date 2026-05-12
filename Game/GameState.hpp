#pragma once

#include <variant>

#include "GameBase.hpp"   // for INVALID_PLAYER, TIE_WINNER

namespace Game {
struct ChanceState {
    static constexpr auto name = "chance";
    bool cardsDealt = false;
    enum Type { DEAL_HOLE, DEAL_FLOP, DEAL_TURN, DEAL_RIVER } type = DEAL_HOLE;
};

struct ActionState {
    static constexpr auto name = "action";
    bool     firstActionTaken = false;
    uint8_t  checksCount      = 0;
    uint32_t currentBet       = 0;
    uint8_t  raiseCount       = 0;
    uint8_t  lastRaiser       = INVALID_PLAYER;
};

struct TerminalState {
    static constexpr auto name = "terminal";
    uint8_t winner = INVALID_PLAYER;
    double  pot    = 0;
    enum Reason { SHOWDOWN, FOLD, ALL_IN } reason = SHOWDOWN;
};

using GameState = std::variant<ChanceState, ActionState, TerminalState>;
}
