#pragma once

#include <functional>
#include <variant>
#include <stdexcept>
#include <algorithm>

#include "BettingConfig.hpp"
#include "GameBase.hpp"
#include "Context/GameContext.hpp"
#include "GameState.hpp"
#include "Utility/CardConversion.hpp"
#include "Utility/Utility.hpp"

namespace Game {

using GameState = GameState;

struct TransitionResult {
    GameState nextState;
    std::function<void(GameContext&)> sideEffect = nullptr;
};

class Transitioner {
public:
    Transitioner() = delete;

    static TransitionResult transition(const Action& action,
                                       const GameContext& context,
                                       const DefaultBettingConfig& config,
                                       const GameState& state) {
        return std::visit([&](const auto& s) {
            return process(action, context, config, s);
        }, state);
    }

private:
    static TransitionResult process(const Action& action, const GameContext& context,
                                    const DefaultBettingConfig& config, const ChanceState& state);
    static TransitionResult process(const Action& action, const GameContext& context,
                                    const DefaultBettingConfig& config, const ActionState& state);
    static TransitionResult process(const Action& action, const GameContext& context,
                                    const DefaultBettingConfig& config, const TerminalState& state);

    static TransitionResult makeTerminal(const GameContext& context, int winner,
                                         double pot, TerminalState::Reason reason);
};

// Implementation
inline TransitionResult Transitioner::process([[maybe_unused]] const Action& action,
                                              [[maybe_unused]] const GameContext& context,
                                              [[maybe_unused]] const DefaultBettingConfig& config,
                                              const ChanceState& state)
{
    TransitionResult result;

    ActionState nextState;
    if (state.type == ChanceState::DEAL_HOLE) {
        // Preflop opens with currentBet = small_blind: SB has already
        // posted (so it owes BB - SB = small_blind to call), and BB
        // is the first to act-or-check.
        nextState = ActionState{.currentBet = context.config().small_blind};
    } else {
        nextState = ActionState{};
    }

    result.nextState = nextState;
    result.sideEffect = [](GameContext& ctx) {
        ctx.setRoundStartingPlayer();
    };

    return result;
}

inline TransitionResult Transitioner::process(const Action& action, const GameContext& context,
                                              [[maybe_unused]] const DefaultBettingConfig& config,
                                              const ActionState& state) {
    return std::visit([&context, &state]<typename ActionType>(const ActionType& act) -> TransitionResult {
        using T = std::decay_t<ActionType>;

        if constexpr (std::is_same_v<T, Check>) {
            if (context.isPreflop() || state.checksCount == NUM_PLAYERS - 1) {
                if (!context.isRiver()) {
                    return {
                        .nextState = ChanceState{},
                        .sideEffect = [](GameContext& ctx) { ctx.advanceRound(); }
                    };
                } else {
                    return makeTerminal(context, -1, context.getPot(), TerminalState::SHOWDOWN);
                }
            } else {
                auto newState = state;
                newState.checksCount++;
                newState.firstActionTaken = true;
                return {
                    .nextState = newState,
                    .sideEffect = [](GameContext& ctx) { ctx.nextPlayer(); }
                };
            }
        }
        else if constexpr (std::is_same_v<T, Raise>) {
            ActionState newState;
            newState.currentBet = act.amount;
            newState.raiseCount = state.raiseCount + 1;
            newState.lastRaiser = context.getCurrentPlayer();
            newState.firstActionTaken = true;

            return {
                .nextState = newState,
                .sideEffect = [amt = act.amount](GameContext& ctx) {
                    ctx.addMoney(ctx.getCurrentPlayer(), amt, false);
                    ctx.nextPlayer();
                }
            };
        }
        else if constexpr (std::is_same_v<T, Fold>) {
            return makeTerminal(context, 1 - context.getCurrentPlayer(), context.getPot(), TerminalState::FOLD);
        }
        else if constexpr (std::is_same_v<T, Call>) {
            if (!state.firstActionTaken) {
                // Handle limp
                auto newState = state;
                newState.firstActionTaken = true;
                return {
                    .nextState = newState,
                    .sideEffect = [amt = act.amount](GameContext& ctx) {
                        ctx.addMoney(ctx.getCurrentPlayer(), amt, true);
                        ctx.nextPlayer();
                    }
                };
            } else {
                uint32_t callerStack = context.getStack(context.getCurrentPlayer());
                uint32_t betToCall = state.currentBet;

                if (context.isPreflop() &&
                    state.raiseCount == 1 &&
                    state.lastRaiser == 1)
                {
                    // SB calling preflop owes BB-SB = small_blind, not
                    // the full BB — its half-blind is already in the pot.
                    betToCall -= context.config().small_blind;
                }

                bool isAllIn = act.amount >= callerStack || act.amount < betToCall;

                if (isAllIn) {
                    double amountToReturn = betToCall - act.amount;
                    const int raiser_player = 1 - context.getCurrentPlayer();

                    if (context.getStack(raiser_player) == 0) {
                        amountToReturn = 0;
                    }

                    auto result = makeTerminal(context, -1, context.getPot() + act.amount, TerminalState::SHOWDOWN);
                    result.sideEffect = [amt = act.amount, amountToReturn](GameContext& ctx) {
                        ctx.addMoney(ctx.getCurrentPlayer(), amt, true);
                        if (amountToReturn > 0) {
                            const int raiser = 1 - ctx.getCurrentPlayer();
                            ctx.updateStack(raiser, amountToReturn);
                            ctx.updatePot(-amountToReturn);
                        }
                    };
                    return result;
                }

                if (!context.isRiver()) {
                    return {
                        .nextState = ChanceState{},
                        .sideEffect = [amt = act.amount](GameContext& ctx) {
                            ctx.addMoney(ctx.getCurrentPlayer(), amt, true);
                            ctx.advanceRound();
                        }
                    };
                } else {
                    auto result = makeTerminal(context, -1, context.getPot() + act.amount, TerminalState::SHOWDOWN);
                    result.sideEffect = [amt = act.amount](GameContext& ctx) {
                        ctx.addMoney(ctx.getCurrentPlayer(), amt, true);
                    };
                    return result;
                }
            }
        } else {
            throw std::logic_error("Invalid action for ActionState");
        }
    }, action);
}

inline TransitionResult Transitioner::process([[maybe_unused]] const Action& action,
                                              [[maybe_unused]] const GameContext& context,
                                              [[maybe_unused]] const DefaultBettingConfig& config,
                                              [[maybe_unused]] const TerminalState& state)
{
    throw std::logic_error("Cannot transition from terminal state");
}

inline TransitionResult Transitioner::makeTerminal(const GameContext& context,
                                                   const int winner,
                                                   double pot,
                                                   const TerminalState::Reason reason)
{
    TerminalState terminal;
    terminal.pot = pot;
    terminal.reason = reason;

    if (reason != TerminalState::SHOWDOWN) {
        terminal.winner = static_cast<uint8_t>(winner);
    } else {
        // Showdown: evaluate both 7-card hands with the TwoPlusTwo lookup
        // and pick the higher rank.
        const auto& raw = context.cards.rawCards;
        const auto cv = [](uint8_t c) { return deck_to_two_plus_two(c); };
        int p0_cards[7] = {
            cv(raw[0]), cv(raw[1]),               // hole
            cv(raw[4]), cv(raw[5]), cv(raw[6]),   // flop
            cv(raw[7]),                           // turn
            cv(raw[8]),                           // river
        };
        int p1_cards[7] = {
            cv(raw[2]), cv(raw[3]),
            cv(raw[4]), cv(raw[5]), cv(raw[6]),
            cv(raw[7]),
            cv(raw[8]),
        };
        const int w = Utility::getWinner(p0_cards, p1_cards);
        // Utility::getWinner returns 0, 1, or 3 (tie); we normalise to
        // {0, 1, TIE_WINNER} so getUtility's tie path picks up cleanly.
        terminal.winner = (w == 3) ? TIE_WINNER : static_cast<uint8_t>(w);
    }
    return {
        .nextState = terminal,
        .sideEffect = nullptr
    };
}

}  // namespace Game