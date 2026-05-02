//
// Compile-time betting configuration. Templated on the pot-fraction menu
// length so a concrete instance can be a constexpr value, suitable for
// `inline constexpr` globals and `if constexpr` gating on its fields.
//
// Pattern: ./PPO_CFR/TexasGame/Utils.hpp — std::array sized by template
// parameter, default-initialised POD fields, constexpr factories.
//
// Notes on the simplification vs. the prior runtime-typed BettingConfig:
//   - DISCRETE_OPTIONS (with std::vector<uint32_t> discreteAmounts),
//     GEOMETRIC_SIZING (with geometricBase / maxGeometricSteps), and
//     CUSTOM_FUNCTION (with std::function<...>) were never wired up to
//     anything that actually called them. They were dead fields blocking
//     constexpr-construction. Dropped here. If/when a non-FIXED_FRACTIONS
//     strategy is needed it can be added back behind an `if constexpr`
//     gate against a new feature flag.
//
// Card encoding & blinds remain delegated to GameConfig / GameContext;
// BettingConfig only describes the abstracted action menu exposed to the
// learner.
//

#ifndef CFR2_BETTINGCONFIG_HPP
#define CFR2_BETTINGCONFIG_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "GameBase.hpp"
#include "GameConfig.hpp"

namespace Game {

// Maximum bet actions emitted by `getAllowedBetSizes` /
// `generateRaiseActions` in a single state. Callers pre-reserve to avoid
// vector reallocations on the hot path.
inline constexpr uint32_t MAX_BET_ACTIONS = 16;

// Strategy tag retained for forward compatibility — a future variant
// (e.g. limit hold'em with discrete bet rungs) can introduce a new enum
// value, gate a `case` block behind `if constexpr (cfg.strategy == X)`,
// and avoid touching the rest of the codebase.
enum class BetSizeStrategy : uint8_t {
    FIXED_FRACTIONS,      // Fixed fractions of pot (only strategy in use)
};

// Templated on the **count** of pot-fraction raise sizes. Default = 11
// matches `Game::DefaultGameConfig` (the 0.25..3.0 menu used by
// main.cpp / config.h).
template <std::size_t NumPotFractions = DEFAULT_NUM_POT_FRACTIONS>
class BettingConfig {
public:
    static constexpr std::size_t kNumPotFractions = NumPotFractions;

    struct Config {
        BetSizeStrategy strategy = BetSizeStrategy::FIXED_FRACTIONS;

        // For FIXED_FRACTIONS — pot multipliers (e.g. {0.5, 1.0, 2.0}).
        std::array<double, NumPotFractions> potFractions{};

        // Global constraints (in mbb). Defaults match
        // `make_nlhe_full_52()`.
        uint32_t minBet            = 1000;
        uint32_t minRaise          = 1000;   // min raise above current bet
        uint8_t  maxRaisesPerRound = 4;
        bool     allowAllIn        = true;
    };

    Config config;

    // Generate allowed bet sizes given current game state (all values in mbb).
    [[nodiscard]] std::vector<uint32_t> getAllowedBetSizes(
        uint32_t pot,
        uint32_t currentBet,
        uint32_t playerStack,
        int      raisesThisRound
    ) const;

    // Generate allowed raise actions (with amounts) given game state.
    [[nodiscard]] std::vector<Action> generateRaiseActions(
        uint32_t pot,
        uint32_t currentBet,
        uint32_t playerStack,
        int      raisesThisRound
    ) const;
};

// ─── Implementation ────────────────────────────────────────────────────

template <std::size_t NumPotFractions>
std::vector<uint32_t> BettingConfig<NumPotFractions>::getAllowedBetSizes(
    const uint32_t pot,
    const uint32_t currentBet,
    const uint32_t playerStack,
    const int      raisesThisRound) const
{
    std::vector<uint32_t> sizes;
    sizes.reserve(MAX_BET_ACTIONS);

    if (raisesThisRound >= config.maxRaisesPerRound) {
        return sizes;
    }

    // Only FIXED_FRACTIONS is currently implemented; the switch is kept
    // (rather than a plain loop) so future strategies can slot in behind
    // `if constexpr` without churning callers.
    switch (config.strategy) {
    case BetSizeStrategy::FIXED_FRACTIONS: {
        for (const double fraction : config.potFractions) {
            const auto size =
                static_cast<uint32_t>(currentBet + (fraction * pot));
            if (size >= config.minBet && size <= playerStack) {
                sizes.push_back(size);
            }
        }
        break;
    }
    }

    if (config.allowAllIn && playerStack > currentBet) {
        if (sizes.empty() || sizes.back() != playerStack) {
            sizes.push_back(playerStack);
        }
    }

    return sizes;
}

template <std::size_t NumPotFractions>
std::vector<Action> BettingConfig<NumPotFractions>::generateRaiseActions(
    const uint32_t pot,
    const uint32_t currentBet,
    const uint32_t playerStack,
    const int      raisesThisRound) const
{
    std::vector<Action> actions;
    actions.reserve(MAX_BET_ACTIONS);

    const auto raiseSizes =
        getAllowedBetSizes(pot, currentBet, playerStack, raisesThisRound);
    for (uint32_t size : raiseSizes) {
        if (size >= playerStack) {
            actions.emplace_back(Raise{size});
            break;
        }
        actions.emplace_back(Raise{size});
    }
    return actions;
}

// ─── Concrete typedef + constexpr factory ──────────────────────────────
//
// Mirrors GameConfig's DefaultGameConfig pattern: downstream code refers
// to `Game::DefaultBettingConfig` so the template parameter stays out of
// the wider type system. Switching variants is done by re-pointing the
// alias and rebuilding.

using DefaultBettingConfig = BettingConfig<DEFAULT_NUM_POT_FRACTIONS>;

// Build a BettingConfig that mirrors `gcfg`'s pot-fraction menu and
// betting structure. `constexpr` so the result can be an `inline
// constexpr` global next to `kPokerConfig`. Keeping the bridge here
// (rather than at the call site in poker_env.cpp) means the runtime
// `.assign()` step is gone — both sides are now `std::array`.
[[nodiscard]] constexpr DefaultBettingConfig make_default_betting_config(
    const DefaultGameConfig& gcfg = kGameConfig) noexcept
{
    DefaultBettingConfig bcfg;
    bcfg.config.strategy           = BetSizeStrategy::FIXED_FRACTIONS;
    bcfg.config.potFractions       = gcfg.pot_fractions;
    bcfg.config.minBet             = gcfg.min_bet;
    bcfg.config.minRaise           = gcfg.min_raise;
    bcfg.config.maxRaisesPerRound  = gcfg.max_raises_per_round;
    bcfg.config.allowAllIn         = gcfg.include_all_in_slot;
    return bcfg;
}

// ─── Live constexpr instance ──────────────────────────────────────────
//
// Pinned to `kGameConfig`'s pot-fraction menu so the two never drift.
// Anything that wants the project's default betting structure should use
// `Game::kBettingConfig` rather than re-running the factory.
inline constexpr DefaultBettingConfig kBettingConfig =
    make_default_betting_config(kGameConfig);

}  // namespace Game

#endif  // CFR2_BETTINGCONFIG_HPP
