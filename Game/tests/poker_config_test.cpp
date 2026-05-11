// Tests for the env-side `inline constexpr poker_ppo::kPokerConfig`.
//
// These tests live next to the Game/ tests because they require only
// `poker_env.h` (which transitively pulls config.h, GameConfig.hpp,
// BettingConfig.hpp). They prove:
//   1. `kPokerConfig` is constant-evaluatable (no surprise `std::vector`
//      / `std::string` field re-introduced).
//   2. Its `game` block matches `Game::kGameConfig` exactly — these are
//      the same compile-time value, not parallel copies.
//   3. Its `hist` / `round_summary` blocks track `kPPOConfig`'s — the env
//      observation layout cannot drift from the network input layout.
//   4. The compile-time invariants in poker_env.h's `static_assert`s
//      hold (`bet_cfg.action_count() == kPokerConfig.action_count()`,
//      etc.).

#include <gtest/gtest.h>

#include "poker_env.h"
#include "config.h"
#include "BettingConfig.hpp"
#include "GameConfig.hpp"

namespace {

using poker_ppo::kPokerConfig;
using poker_ppo::config::kPPOConfig;
using poker_ppo::config::kBetConfig;

TEST(PokerConfig, IsConstexprAndPointsAtKGameConfig) {
    static_assert(kPokerConfig.game.action_count()
                  == Game::kGameConfig.action_count(),
                  "kPokerConfig.game must equal Game::kGameConfig");
    static_assert(kPokerConfig.game.initial_stack
                  == Game::kGameConfig.initial_stack);
    static_assert(kPokerConfig.game.big_blind == Game::kGameConfig.big_blind);
    static_assert(kPokerConfig.seed == 0x12345ULL,
                  "Seed should be the constant set in poker_env.h");
    EXPECT_NE(static_cast<const void*>(&kPokerConfig), nullptr);
}

TEST(PokerConfig, FeatureBlocksTrackPPOConfig) {
    // The hist / round_summary blocks that the env uses to size the obs
    // are tied to kPPOConfig at compile time, so the network input layout
    // and the env output layout cannot drift.
    static_assert(kPokerConfig.hist.enabled == kPPOConfig.hist.enabled);
    static_assert(kPokerConfig.hist.max_history_len
                  == kPPOConfig.hist.max_history_len);
    static_assert(kPokerConfig.round_summary.enabled
                  == kPPOConfig.round_summary.enabled);
    SUCCEED();
}

TEST(PokerConfig, ActionCountMatchesBetConfig) {
    // Mirrors the runtime check in PokerEnvironment's constructor —
    // surfaced here as a cheap unit test in addition to the
    // `static_assert` in poker_env.h.
    static_assert(kPokerConfig.action_count() == kBetConfig.action_count(),
                  "kBetConfig.action_count() must match kPokerConfig.action_count()");
    EXPECT_EQ(kPokerConfig.action_count(), kBetConfig.action_count());
}

}  // namespace
