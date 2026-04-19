//
// Created by Elijah Crain on 12/21/25.
//

#ifndef CFR2_ACTIONPOLICY_HPP
#define CFR2_ACTIONPOLICY_HPP

#include <algorithm>
#include <vector>
#include "GameBase.hpp"
#include "Context/GameContext.hpp"
#include "GameState.hpp"
#include "BettingConfig.hpp"

namespace Game {

// Bounds structure for continuous action space
struct ActionBounds {
    uint32_t minRaise = 0;
    uint32_t maxRaise = 0;
    uint32_t callAmount = 0;
    bool canCheck = false;
    bool canFold = false;
    bool canRaise = false;
    bool canCall = false;
    bool isChance = false;
    bool isTerminal = false;

    [[nodiscard]] bool hasActions() const noexcept {
        return isChance || canCheck || canFold || canRaise || canCall;
    }

    [[nodiscard]] size_t numDiscreteActions() const noexcept {
        // For algorithms that need action count: fold/call/check + 1 continuous raise
        return static_cast<size_t>(canFold) +
               static_cast<size_t>(canCall) +
               static_cast<size_t>(canCheck) +
               static_cast<size_t>(canRaise);
    }
};

// ============================================================================
// Discrete Action Policy - uses explicit list of actions
// ============================================================================
struct DiscreteActionPolicy {
    using ActionSet = std::vector<Action>;

    [[nodiscard]] static bool isValidAction(const Action& action, const ActionSet& available) noexcept {
        return std::ranges::any_of(available,
            [&action](const Action& avail) {
                return std::visit([](const auto& a1, const auto& a2) {
                    using T1 = std::decay_t<decltype(a1)>;
                    using T2 = std::decay_t<decltype(a2)>;
                    if constexpr (!std::is_same_v<T1, T2>) {
                        return false;
                    } else if constexpr (std::is_same_v<T1, Raise> || std::is_same_v<T1, Call>) {
                        auto diff = a1.amount > a2.amount ? a1.amount - a2.amount : a2.amount - a1.amount;
                        return diff < 1;
                    } else {
                        return true;
                    }
                }, action, avail);
            });
    }

    [[nodiscard]] static ActionSet makeChanceActionSet() {
        return {Chance{}};
    }

    [[nodiscard]] static ActionSet makeEmptyActionSet() {
        return {};
    }

    [[nodiscard]] static ActionSet generateActions(
        const BettingConfig& config,
        const GameContext& context,
        const ActionState& state);

    // Utility: get number of available actions
    [[nodiscard]] static size_t numActions(const ActionSet& actions) noexcept {
        return actions.size();
    }
};

// ============================================================================
// Continuous Action Policy - uses bounds for raise amounts
// ============================================================================
struct ContinuousActionPolicy {
    using ActionSet = ActionBounds;

    [[nodiscard]] static bool isValidAction(const Action& action, const ActionSet& bounds) noexcept {
        return std::visit([&bounds](const auto& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Chance>) {
                return bounds.isChance;
            } else if constexpr (std::is_same_v<T, Fold>) {
                return bounds.canFold;
            } else if constexpr (std::is_same_v<T, Check>) {
                return bounds.canCheck;
            } else if constexpr (std::is_same_v<T, Call>) {
                return bounds.canCall && a.amount == bounds.callAmount;
            } else if constexpr (std::is_same_v<T, Raise>) {
                return bounds.canRaise &&
                       a.amount >= bounds.minRaise &&
                       a.amount <= bounds.maxRaise;
            } else {
                return false;
            }
        }, action);
    }

    [[nodiscard]] static ActionSet makeChanceActionSet() {
        ActionBounds bounds;
        bounds.isChance = true;
        return bounds;
    }

    [[nodiscard]] static ActionSet makeEmptyActionSet() {
        ActionBounds bounds;
        bounds.isTerminal = true;
        return bounds;
    }

    [[nodiscard]] static ActionSet generateActions(
        const BettingConfig& config,
        const GameContext& context,
        const ActionState& state);

    // Utility: get number of discrete action types (not including continuous range)
    [[nodiscard]] static size_t numActions(const ActionSet& bounds) noexcept {
        return bounds.numDiscreteActions();
    }
};

// ============================================================================
// Implementation: DiscreteActionPolicy::generateActions
// ============================================================================
inline DiscreteActionPolicy::ActionSet DiscreteActionPolicy::generateActions(
    const BettingConfig& config,
    const GameContext& context,
    const ActionState& state)
{
    std::vector<Action> actions;
    actions.reserve(MAX_BET_ACTIONS);

    const int current_player = context.players.currentPlayer;

    if ((context.round.isPreflop() && !state.firstActionTaken) || state.raiseCount > 0) {
        actions = config.generateRaiseActions(
            context.betting.pot,
            state.currentBet,
            context.players.stacks[current_player],
            state.raiseCount
        );

        uint32_t callAmount = state.currentBet;
        if (context.round.isPreflop() &&
            state.raiseCount == 1 &&
            state.lastRaiser == 1)
        {
            callAmount -= 500;
        }

        uint32_t finalCallAmount = std::min(callAmount, context.players.stacks[current_player]);
        actions.emplace_back(Call{finalCallAmount});
        actions.emplace_back(Fold{});
    } else {
        actions = config.generateRaiseActions(
            context.betting.pot,
            state.currentBet,
            context.players.stacks[current_player],
            state.raiseCount
        );
        actions.emplace_back(Check{});
    }

    return actions;
}

// ============================================================================
// Implementation: ContinuousActionPolicy::generateActions
// ============================================================================
inline ContinuousActionPolicy::ActionSet ContinuousActionPolicy::generateActions(
    const BettingConfig& config,
    const GameContext& context,
    const ActionState& state)
{
    ActionBounds bounds;
    const int current_player = context.players.currentPlayer;
    const uint32_t playerStack = context.players.stacks[current_player];

    // Determine if we're facing a bet/raise
    const bool facingBet = (context.round.isPreflop() && !state.firstActionTaken) || state.raiseCount > 0;

    if (facingBet) {
        bounds.canFold = true;
        bounds.canCall = true;

        uint32_t callAmount = state.currentBet;
        if (context.round.isPreflop() &&
            state.raiseCount == 1 &&
            state.lastRaiser == 1)
        {
            callAmount -= 500;
        }
        bounds.callAmount = std::min(callAmount, playerStack);
    } else {
        bounds.canCheck = true;
    }

    // Raise bounds
    if (state.raiseCount < config.config.maxRaisesPerRound && playerStack > state.currentBet) {
        // Minimum raise: current bet + min raise increment
        uint32_t minRaiseTotal = state.currentBet + config.config.minRaise;
        uint32_t minRaise = std::max(minRaiseTotal, config.config.minBet);
        minRaise = std::min(minRaise, playerStack);

        // Maximum raise: player's stack (all-in)
        uint32_t maxRaise = playerStack;

        // Only enable raise if min <= max
        if (minRaise <= maxRaise) {
            bounds.canRaise = true;
            bounds.minRaise = minRaise;
            bounds.maxRaise = maxRaise;
        }
    }

    return bounds;
}

}  // namespace Game

#endif //CFR2_ACTIONPOLICY_HPP