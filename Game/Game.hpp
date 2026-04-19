//
// Created by Elijah Crain on 10/5/25.
//

#ifndef CFR2_VARIANTTEXAS_GAME_HPP
#define CFR2_VARIANTTEXAS_GAME_HPP
#include <random>
#include <stdexcept>
#include <algorithm>

#include "BettingConfig.hpp"
#include "Transitioner.hpp"
#include "GameBase.hpp"
#include "GameConfig.hpp"
#include "GameState.hpp"
#include "Context/GameContext.hpp"
#include "ActionPolicy.hpp"

namespace Game {

template <typename ActionPolicy = DiscreteActionPolicy>
class Game {
public:
using ChanceAction = Chance;
using InfoSet = uint64_t;
using ActionSet = typename ActionPolicy::ActionSet;

static constexpr int PlayerNum = 2;

// Config is copied into the Game so GameContext can hold a stable pointer.
explicit Game(std::mt19937& engine, GameConfig game_cfg = {}, BettingConfig betting_cfg = {})
    : m_rng(engine),
      m_game_config(std::move(game_cfg)),
      m_betting_config(std::move(betting_cfg)),
      m_context(m_game_config) {
    m_current_state = ChanceState{};
    m_available_actions = ActionPolicy::makeChanceActionSet();
    m_context.initializeCards(m_rng);
}

// Process any action type
void transition(const Action& action) {
    // Validate action
    if (!isValidAction(action)) {
        throw std::invalid_argument("Invalid action for current state");
    }

    auto [nextState, sideEffect] =
        Transitioner::transition(action, m_context, m_betting_config, m_current_state);

    updateInfoSets(action);

    // Execute side effects
    if (sideEffect) {
        sideEffect(m_context);
    }

    // Update state
    m_current_state = nextState;

    // Generate available actions using policy
    updateAvailableActions();
}

// Reset game to initial state
void reInitialize() {
    m_context.reset(m_rng);
    m_current_state = ChanceState{};
    m_available_actions = ActionPolicy::makeChanceActionSet();
}

// Get available actions with their parameters
[[nodiscard]] const ActionSet& getActions() const noexcept {
    return m_available_actions;
}

// For continuous policy: get bounds directly
[[nodiscard]] ActionBounds getActionBounds() const noexcept {
    if constexpr (std::is_same_v<ActionPolicy, ContinuousActionPolicy>) {
        return m_available_actions;
    } else {
        // Convert discrete actions to bounds
        return discreteToBounds(m_available_actions);
    }
}

// State queries
[[nodiscard]] std::string getType() const {
    return std::visit([](const auto& state) {
        return std::string(state.name);
    }, m_current_state);
}

[[nodiscard]] bool isTerminal() const noexcept {
    return std::holds_alternative<TerminalState>(m_current_state);
}

[[nodiscard]] int32_t getUtility(int player) const {
    if (const auto* terminal = std::get_if<TerminalState>(&m_current_state)) {
        const auto contribution = static_cast<int32_t>(
            m_game_config.initial_stack - m_context.getStack(player));
        if (terminal->winner == player) {
            return static_cast<int32_t>(m_context.getPot()) - contribution;
        }
        if (terminal->winner == -1) {  // Draw
            return static_cast<int32_t>(m_context.getPot() / 2) - contribution;
        }
        // Lose
        return -contribution;
    }
    return 0;
}

// Getters
[[nodiscard]] int getCurrentPlayer() const noexcept { return m_context.getCurrentPlayer(); }
[[nodiscard]] int getCurrentRound() const noexcept { return m_context.getRoundNumber(); }

[[nodiscard]] std::string getInfoSetString(int player) const {
    return m_context.getInfoSetString(player);
}
[[nodiscard]] uint64_t getInfoSet(int player) const noexcept {
    return m_context.getInfoSetNumericId(player);
}
[[nodiscard]] const InfoSetData& getInfoSetData(int player) const noexcept {
    return m_context.getInfoSetData(player);
}

[[nodiscard]] const GameContext& getContext() const noexcept { return m_context; }

[[nodiscard]] std::vector<uint32_t> getAbstractedActionSizes() const {
    uint32_t pot = m_context.getPot();
    uint32_t currentBet = getCurrentBet();
    uint32_t stack = m_context.getStack(m_context.getCurrentPlayer());
    int raises = getRaisesThisRound();

    return m_betting_config.getAllowedBetSizes(pot, currentBet, stack, raises);
}

[[nodiscard]] uint32_t getCurrentBet() const noexcept {
    if (const auto* betState = std::get_if<ActionState>(&m_current_state)) {
        return betState->currentBet;
    }
    return 0;
}

// Get number of available actions
[[nodiscard]] size_t numActions() const noexcept {
    return ActionPolicy::numActions(m_available_actions);
}

// Static factory methods for common configurations
static BettingConfig makeFixedLimitConfig(uint32_t smallBet, uint32_t bigBet) {
    BettingConfig config;
    config.config.strategy = BettingConfig::BetSizeStrategy::DISCRETE_OPTIONS;
    config.config.discreteAmounts = {smallBet, bigBet};
    config.config.maxRaisesPerRound = 4;
    return config;
}

static BettingConfig makePotLimitConfig() {
    BettingConfig config;
    config.config.strategy = BettingConfig::BetSizeStrategy::FIXED_FRACTIONS;
    config.config.potFractions = {1.0f};
    config.config.maxRaisesPerRound = 999;
    return config;
}

static BettingConfig makeNoLimitConfig(std::vector<double> preferredSizes = {0.5f, 0.75f, 1.0f}) {
    BettingConfig config;
    config.config.strategy = BettingConfig::BetSizeStrategy::FIXED_FRACTIONS;
    config.config.potFractions = std::move(preferredSizes);
    config.config.allowAllIn = true;
    config.config.maxRaisesPerRound = 999;
    return config;
}

[[nodiscard]] const GameConfig& getGameConfig() const noexcept { return m_game_config; }

private:
std::mt19937& m_rng;
GameConfig m_game_config;          // owned copy; referenced by m_context
BettingConfig m_betting_config;
GameContext m_context;             // must be declared after m_game_config
GameState m_current_state;
ActionSet m_available_actions;

[[nodiscard]] bool isValidAction(const Action& action) const noexcept {
    return ActionPolicy::isValidAction(action, m_available_actions);
}

void updateAvailableActions() {
    if (std::holds_alternative<TerminalState>(m_current_state)) {
        m_available_actions = ActionPolicy::makeEmptyActionSet();
    } else if (std::holds_alternative<ChanceState>(m_current_state)) {
        m_available_actions = ActionPolicy::makeChanceActionSet();
    } else if (const auto* actionState = std::get_if<ActionState>(&m_current_state)) {
        m_available_actions = ActionPolicy::generateActions(
            m_betting_config, m_context, *actionState);
    }
}

void updateInfoSets(const Action& action) {
    std::visit([this]<typename ActionType>(const ActionType& specific_action) {
        using T = std::decay_t<ActionType>;
        if constexpr (std::is_same_v<T, Chance>) {
            const int currentRound = m_context.getRoundNumber();
            for (int i = 0; i < NUM_PLAYERS; ++i) {
                m_context.setCardHashFromDeck(i, currentRound);
            }
        } else if constexpr (std::is_same_v<T, Fold>) {
            // Fold leads to terminal - no need to record
        } else {
            const int currentRound = m_context.getRoundNumber();
            const uint32_t actionValue = InfoSetData::actionToValue(specific_action);
            for (int i = 0; i < NUM_PLAYERS; ++i) {
                m_context.addInfoSetAction(i, currentRound, actionValue);
            }
        }
    }, action);
}

[[nodiscard]] int getRaisesThisRound() const noexcept {
    return m_context.getRaiseNum();
}

// Helper to convert discrete actions to bounds (for getActionBounds compatibility)
[[nodiscard]] static ActionBounds discreteToBounds(const std::vector<Action>& actions) {
    ActionBounds bounds;
    for (const auto& action : actions) {
        std::visit([&bounds](const auto& a) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, Chance>) {
                bounds.isChance = true;
            } else if constexpr (std::is_same_v<T, Fold>) {
                bounds.canFold = true;
            } else if constexpr (std::is_same_v<T, Check>) {
                bounds.canCheck = true;
            } else if constexpr (std::is_same_v<T, Call>) {
                bounds.canCall = true;
                bounds.callAmount = a.amount;
            } else if constexpr (std::is_same_v<T, Raise>) {
                bounds.canRaise = true;
                bounds.maxRaise = std::max(bounds.maxRaise, a.amount);
                if (bounds.minRaise == 0) {
                    bounds.minRaise = a.amount;
                } else {
                    bounds.minRaise = std::min(bounds.minRaise, a.amount);
                }
            }
        }, action);
    }
    return bounds;
}
};

// Type aliases for convenience
using DiscreteGame = Game<DiscreteActionPolicy>;
using ContinuousGame = Game<ContinuousActionPolicy>;

}  // namespace Game

#endif  // CFR2_VARIANTTEXAS_GAME_HPP