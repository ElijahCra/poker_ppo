//
// Created by Elijah Crain on 10/5/25.
//

#ifndef CFR2_BETTINGCONFIG_HPP
#define CFR2_BETTINGCONFIG_HPP
#include <functional>
#include <vector>
#include <cstdint>

#include "GameBase.hpp"

namespace Game {
static constexpr uint32_t MAX_BET_ACTIONS = 8;
class BettingConfig {
public:
    // Different strategies for generating bet sizes
    enum class BetSizeStrategy {
        FIXED_FRACTIONS,      // Fixed fractions of pot
        GEOMETRIC_SIZING,     // Geometric progression
        CUSTOM_FUNCTION,      // Custom function
        DISCRETE_OPTIONS      // Specific discrete options
    };

    struct Config {
        BetSizeStrategy strategy = BetSizeStrategy::DISCRETE_OPTIONS;

        // For FIXED_FRACTIONS (e.g., {0.33, 0.5, 0.75, 1.0} for 33%, 50%, 75%, 100% pot)
        std::vector<double> potFractions = {0.5, 0.75, 1.0, 1.5};

        // For GEOMETRIC_SIZING
        double geometricBase = 2.0;
        int maxGeometricSteps = 3;

        // For DISCRETE_OPTIONS (absolute amounts in mbb)
        std::vector<uint32_t> discreteAmounts = {1000, 2000, 3000, 5000};

        // For CUSTOM_FUNCTION
        std::function<std::vector<uint32_t>(uint32_t pot, uint32_t currentBet, uint32_t stack)> customFunction;

        // Global constraints (in mbb)
        uint32_t minBet = 1000;
        uint32_t minRaise = 1000;  // Min raise over current bet
        int maxRaisesPerRound = 4;
        bool allowAllIn = true;
    };

    Config config;

    // Generate allowed bet sizes given current game state (all values in mbb)
    [[nodiscard]] std::vector<uint32_t> getAllowedBetSizes(
        uint32_t pot,
        uint32_t currentBet,
        uint32_t playerStack,
        int raisesThisRound
    ) const;

    // Generate allowed actions (with amounts) given game state
    [[nodiscard]] std::vector<Action> generateRaiseActions(
        uint32_t pot,
        uint32_t currentBet,
        uint32_t playerStack,
        int raisesThisRound
    ) const;
};

// Implementation
inline std::vector<uint32_t> BettingConfig::getAllowedBetSizes(
    const uint32_t pot,
    const uint32_t currentBet,
    const uint32_t playerStack,
    const int raisesThisRound
) const {

    std::vector<uint32_t> sizes;
    sizes.reserve(MAX_BET_ACTIONS);  // Reserve space to avoid reallocations

    // Check if we can still raise
    if (raisesThisRound >= config.maxRaisesPerRound) {
        return sizes;  // No more raises allowed
    }

    switch (config.strategy) {
    case BetSizeStrategy::FIXED_FRACTIONS: {
            for (const double fraction : config.potFractions) {
                auto size = static_cast<uint32_t>(currentBet + (fraction * pot));
                if (size >= config.minBet && size <= playerStack) {
                    sizes.push_back(size);
                }
            }
            break;
    }

    case BetSizeStrategy::GEOMETRIC_SIZING: {
            auto size = static_cast<double>(currentBet);
            for (int i = 0; i < config.maxGeometricSteps; ++i) {
                size *= config.geometricBase;
                if (static_cast<uint32_t>(size) <= playerStack) {
                    sizes.push_back(static_cast<uint32_t>(size));
                } else {
                    break;
                }
            }
            break;
    }

    case BetSizeStrategy::DISCRETE_OPTIONS: {
            for (uint32_t amount : config.discreteAmounts) {
                if (amount > currentBet && amount <= playerStack) {
                    sizes.push_back(amount);
                }
            }
            break;
    }

    case BetSizeStrategy::CUSTOM_FUNCTION: {
            if (config.customFunction) {
                sizes = config.customFunction(pot, currentBet, playerStack);
            }
            break;
    }
    }

    // Add all-in option if allowed and it's different from other sizes
    if (config.allowAllIn && playerStack > currentBet) {
        if (sizes.empty() || sizes.back() != playerStack) {
            sizes.push_back(playerStack);
        }
    }

    return sizes;
}

inline std::vector<Action> BettingConfig::generateRaiseActions(
    const uint32_t pot,
    const uint32_t currentBet,
    const uint32_t playerStack,
    const int raisesThisRound
) const {

    std::vector<Action> actions;
    actions.reserve(MAX_BET_ACTIONS);  // Reserve space to avoid reallocations

    // Get raise sizes
    const auto raiseSizes = getAllowedBetSizes(pot, currentBet, playerStack, raisesThisRound);
    for (uint32_t size : raiseSizes) {
        if (size >= playerStack) {
            actions.emplace_back(Raise{size});
            break;
        }
        actions.emplace_back(Raise{size});
    }
    return actions;
}
}
#endif //CFR2_BETTINGCONFIG_HPP