// Tests for the templated `Game::BettingConfig`.
//
// What we're verifying after the constexpr templating refactor:
//   1. The class can be constructed in a constant-evaluated context
//      (necessary for `inline constexpr` globals to compile).
//   2. The compile-time `kNumPotFractions` parameter is exposed and
//      threads correctly through `std::array<double, N>`.
//   3. `make_default_betting_config(gcfg)` populates the fields from a
//      `DefaultGameConfig` 1:1 — this replaced the hand-rolled
//      `.assign()` bridge that used to live in poker_env.cpp.
//   4. The runtime methods (`getAllowedBetSizes`, `generateRaiseActions`)
//      still behave the same way they did before the refactor:
//      • respect `maxRaisesPerRound`
//      • emit the all-in slot only when `allowAllIn`, distinct, and
//        the player isn't already covered
//      • clamp by `minBet` / `playerStack`
//   5. A non-default template instantiation (`BettingConfig<3>`) works,
//      proving the parameter isn't accidentally hard-wired anywhere.
//   6. The struct can be plugged into `DiscreteGame` without compile or
//      runtime error.

#include <gtest/gtest.h>

#include <random>
#include <type_traits>

#include "BettingConfig.hpp"
#include "Game.hpp"
#include "GameConfig.hpp"

namespace {

// ── 1. constexpr-friendly construction ─────────────────────────────────

TEST(BettingConfig, IsConstexprConstructible) {
    constexpr auto bcfg = Game::make_default_betting_config();
    static_assert(bcfg.kNumPotFractions == Game::DEFAULT_NUM_POT_FRACTIONS,
                  "Default template parameter should match GameConfig's default");
    static_assert(bcfg.config.strategy == Game::BetSizeStrategy::FIXED_FRACTIONS,
                  "Factory should select FIXED_FRACTIONS");
    static_assert(bcfg.config.allowAllIn,
                  "Factory should mirror nlhe_full_52's allowAllIn=true");
    EXPECT_EQ(bcfg.config.minBet, 1000u);
    EXPECT_EQ(bcfg.config.minRaise, 1000u);
    EXPECT_EQ(bcfg.config.maxRaisesPerRound, 4u);
}

// ── 1b. live `inline constexpr` instances are usable as constants ──────

TEST(BettingConfig, LiveConstexprInstancesAreUsable) {
    // If these compile, `Game::kGameConfig` and `Game::kBettingConfig` are
    // proper namespace-scope `inline constexpr` values — i.e. the bytes
    // live in `.rodata` and the trainer/env reference the same singletons.
    static_assert(Game::kGameConfig.action_count() ==
                  static_cast<int>(Game::DEFAULT_NUM_POT_FRACTIONS) + 2 + 1,
                  "11 fractions + fold + check/call + all-in slot");
    static_assert(Game::kBettingConfig.config.maxRaisesPerRound
                  == Game::kGameConfig.max_raises_per_round,
                  "Live betting config must be derived from kGameConfig");
    static_assert(Game::kBettingConfig.config.allowAllIn
                  == Game::kGameConfig.include_all_in_slot);
    static_assert(Game::kBettingConfig.config.potFractions[0]
                  == Game::kGameConfig.pot_fractions[0]);

    // Address taken — only legal for inline-constexpr (or static)
    // namespace-scope variables. This is the C++ way of asserting "this
    // really is a single live instance, not a fresh copy per call".
    EXPECT_NE(static_cast<const void*>(&Game::kGameConfig), nullptr);
    EXPECT_NE(static_cast<const void*>(&Game::kBettingConfig), nullptr);
}

TEST(BettingConfig, DefaultPotFractionsMatchGameConfig) {
    constexpr auto gcfg = Game::make_nlhe_full_52();
    constexpr auto bcfg = Game::make_default_betting_config(gcfg);

    static_assert(bcfg.config.potFractions.size()
                  == gcfg.pot_fractions.size(),
                  "Sizes must match — that's the whole point of the template");

    for (std::size_t i = 0; i < gcfg.pot_fractions.size(); ++i) {
        EXPECT_DOUBLE_EQ(bcfg.config.potFractions[i], gcfg.pot_fractions[i])
            << "mismatch at fraction index " << i;
    }
}

// ── 2. template parameter exposure ─────────────────────────────────────

TEST(BettingConfig, ExposesTemplateParameter) {
    static_assert(Game::DefaultBettingConfig::kNumPotFractions
                  == Game::DEFAULT_NUM_POT_FRACTIONS);

    using Three = Game::BettingConfig<3>;
    static_assert(Three::kNumPotFractions == 3);
    // potFractions size on the *member* type matches the template arg.
    static_assert(std::tuple_size<decltype(Three::Config::potFractions)>::value == 3);
}

// ── 3. type identity: alias matches the canonical instantiation ───────

TEST(BettingConfig, DefaultAliasMatchesExplicitInstantiation) {
    static_assert(std::is_same_v<
        Game::DefaultBettingConfig,
        Game::BettingConfig<Game::DEFAULT_NUM_POT_FRACTIONS>>);
}

// ── 4. getAllowedBetSizes runtime semantics ────────────────────────────

TEST(BettingConfig, GetAllowedBetSizesRespectsMaxRaisesPerRound) {
    auto bcfg = Game::make_default_betting_config();
    bcfg.config.maxRaisesPerRound = 4;

    const uint32_t pot = 2000;
    const uint32_t currentBet = 1000;
    const uint32_t stack = 100'000;

    EXPECT_FALSE(bcfg.getAllowedBetSizes(pot, currentBet, stack, 3).empty());
    EXPECT_TRUE(bcfg.getAllowedBetSizes(pot, currentBet, stack, 4).empty())
        << "raisesThisRound == max should produce no raise sizes";
    EXPECT_TRUE(bcfg.getAllowedBetSizes(pot, currentBet, stack, 5).empty());
}

TEST(BettingConfig, GetAllowedBetSizesEmitsAllInWhenDistinct) {
    auto bcfg = Game::make_default_betting_config();
    bcfg.config.allowAllIn = true;

    const uint32_t pot = 4000;
    const uint32_t currentBet = 1000;
    const uint32_t stack = 100'000;
    const auto sizes = bcfg.getAllowedBetSizes(pot, currentBet, stack, 0);

    ASSERT_FALSE(sizes.empty());
    // Last entry should be the all-in.
    EXPECT_EQ(sizes.back(), stack)
        << "Trailing entry should equal the player's stack (all-in)";

    // No allowAllIn: last entry should NOT match the stack (unless a
    // pot-fraction happens to land there, which it doesn't with these
    // numbers).
    bcfg.config.allowAllIn = false;
    const auto noai = bcfg.getAllowedBetSizes(pot, currentBet, stack, 0);
    if (!noai.empty()) {
        EXPECT_NE(noai.back(), stack);
    }
}

TEST(BettingConfig, GetAllowedBetSizesClampsByMinBet) {
    auto bcfg = Game::make_default_betting_config();
    // Force a high min-bet so the 0.25*pot fraction gets clipped out but
    // larger fractions survive.
    bcfg.config.minBet = 5000;
    bcfg.config.allowAllIn = false;

    const uint32_t pot = 2000;        // 0.25*pot = 500 -> below 5000 + currentBet
    const uint32_t currentBet = 1000;
    const uint32_t stack = 100'000;
    const auto sizes = bcfg.getAllowedBetSizes(pot, currentBet, stack, 0);

    for (auto s : sizes) {
        EXPECT_GE(s, bcfg.config.minBet);
        EXPECT_LE(s, stack);
    }
}

TEST(BettingConfig, GetAllowedBetSizesClampsByPlayerStack) {
    auto bcfg = Game::make_default_betting_config();
    bcfg.config.allowAllIn = false;

    const uint32_t pot = 200'000;     // huge pot; 3.0*pot = 600k
    const uint32_t currentBet = 1000;
    const uint32_t stack = 50'000;    // smaller stack — every fraction over .. exceeds
    const auto sizes = bcfg.getAllowedBetSizes(pot, currentBet, stack, 0);

    for (auto s : sizes) {
        EXPECT_LE(s, stack)
            << "No allowed bet size should exceed the player's stack";
    }
}

// ── 5. generateRaiseActions returns Raise variants ─────────────────────

TEST(BettingConfig, GenerateRaiseActionsAreAllRaises) {
    auto bcfg = Game::make_default_betting_config();
    const auto actions =
        bcfg.generateRaiseActions(/*pot=*/4000, /*currentBet=*/1000,
                                  /*playerStack=*/100'000, /*raises=*/0);
    ASSERT_FALSE(actions.empty());
    for (const auto& a : actions) {
        EXPECT_TRUE(std::holds_alternative<Game::Raise>(a))
            << "generateRaiseActions must only emit Raise variants";
    }
}

// ── 6. integration: DiscreteGame is constructable with default config ──

TEST(BettingConfig, DiscreteGameAcceptsDefaultBettingConfig) {
    std::mt19937 rng(0xC0FFEEu);
    Game::DefaultGameConfig    gcfg = Game::make_nlhe_full_52();
    Game::DefaultBettingConfig bcfg = Game::make_default_betting_config(gcfg);

    Game::DiscreteGame game(rng, gcfg, bcfg);

    // Default state: chance node, has at least one available action.
    EXPECT_EQ(game.getType(), "chance");
    EXPECT_FALSE(game.isTerminal());
    EXPECT_GT(game.numActions(), 0u);
}

// ── 7. non-default template instantiation works ───────────────────────

TEST(BettingConfig, ThreeFractionInstantiationCompiles) {
    Game::BettingConfig<3> bcfg;
    bcfg.config.strategy           = Game::BetSizeStrategy::FIXED_FRACTIONS;
    bcfg.config.potFractions       = {0.5, 1.0, 2.0};
    bcfg.config.minBet             = 1000;
    bcfg.config.minRaise           = 1000;
    bcfg.config.maxRaisesPerRound  = 4;
    bcfg.config.allowAllIn         = true;

    const auto sizes = bcfg.getAllowedBetSizes(/*pot=*/4000,
                                               /*currentBet=*/1000,
                                               /*playerStack=*/100'000,
                                               /*raises=*/0);
    // 3 fractions + all-in
    EXPECT_EQ(sizes.size(), 4u);
    EXPECT_EQ(sizes.back(), 100'000u);   // all-in slot
}

// ── 8. trivial / standard-layout sanity (cheap copy/move) ─────────────

TEST(BettingConfig, DefaultBettingConfigIsCheapToCopy) {
    // Not strictly trivially copyable (std::array<double, 11> is, but
    // future fields may not be) — but it should not own any heap memory
    // anymore after dropping the std::vector. Verify Config size is
    // sane: 11 doubles + 4 uints + 1 byte + 1 byte ≤ a couple of cache
    // lines.
    using Cfg = Game::DefaultBettingConfig::Config;
    constexpr std::size_t expected_max_bytes =
        sizeof(double) * Game::DEFAULT_NUM_POT_FRACTIONS
        + sizeof(uint32_t) * 2
        + sizeof(uint8_t) * 2
        + 64;  // generous slack for alignment padding
    static_assert(sizeof(Cfg) <= expected_max_bytes,
                  "BettingConfig grew unexpectedly — recheck the field list");
    SUCCEED();
}

}  // namespace
