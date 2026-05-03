#pragma once

#include "config.h"
#include "environment.h"
#include "observation_builder.h"
#include "Game.hpp"
#include "GameConfig.hpp"

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// Poker environment adapter
// ─────────────────────────────────────────────────────────────────────────────
//
// Wraps Game::DiscreteGame (heads-up no-limit hold'em from the Poker project) in
// the IPokerEnvironment interface used by PPO.
//
// Action space (indices into the PPO policy's categorical):
//   0                       → Fold
//   1                       → Check or Call (whichever is legal)
//   2 .. 1+K                → K pot-fraction raises defined in PokerConfig
//   2+K  (if allow_all_in)  → All-in raise (only exposed when distinct)
//
// The adapter auto-advances through ChanceState transitions (which in the
// Poker implementation just flip the round marker; all cards are dealt at hand
// start).  PPO therefore only ever sees ActionState or terminal snapshots.
//
// Reward: the terminal reward is the utility delta for seat 0 (in "mbb" /
// millibigblinds as stored by Poker), normalised by INITIAL_STACK so it sits in
// a reasonable range for the critic.  PPO handles the sign-flip for seat 1.

// Wraps a Game::DefaultGameConfig (the poker variant) plus PPO-side knobs.
// All poker-rule tuning lives in `game`; only per-run seeding and adapter
// flags belong on PokerConfig itself.
struct PokerConfig {
    ::Game::DefaultGameConfig  game{};           // deck, stacks, blinds, betting structure
    BetHistoryConfig    hist{};           // attention-encoder layout (T, F, ...)
    RoundSummaryConfig  round_summary{};  // per-round summary feature block

    // Seed used as a base — each created env gets seed ^ instance hash.
    uint64_t seed = 0x9E3779B97F4A7C15ull;

    [[nodiscard]] constexpr int num_raise_slots() const noexcept { return game.num_raise_slots(); }
    [[nodiscard]] constexpr int action_count()    const noexcept { return game.action_count(); }
};

// ─── Live constexpr instance ──────────────────────────────────────────
//
// Single compile-time PokerConfig consumed by PokerEnvironment /
// PokerEnvironmentFactory / main.cpp. The trainer-side feature gates
// (`hist`, `round_summary`) are pulled directly from `kPPOConfig` so the
// env's observation layout and the network's input layout can never
// drift apart. The seed is the PRNG base — each env XORs an instance
// hash on top.
inline constexpr PokerConfig kPokerConfig{
    .game          = ::Game::kGameConfig,
    .hist          = config::kPPOConfig.hist,
    .round_summary = config::kPPOConfig.round_summary,
    .seed          = 0x12345ULL,
};

// Compile-time invariants. These mirror the runtime check in the
// PokerEnvironment constructor — having them as static_asserts means
// any drift between BetConfig and the game's action layout fails to
// build instead of throwing at runtime.
static_assert(kPokerConfig.action_count() == config::kBetConfig.action_count(),
              "kBetConfig.action_count() must match kPokerConfig.action_count() "
              "(2 + num_raise_slots)");
static_assert(kPokerConfig.game.max_raises_per_round
              == static_cast<uint8_t>(config::kBetConfig.max_bets_per_round),
              "BetConfig.max_bets_per_round must match game.max_raises_per_round");

class PokerEnvironment : public IPokerEnvironment {
public:
    PokerEnvironment(const PokerConfig& poker_cfg,
                        const BetConfig& bet_cfg,
                        uint64_t seed);

    int obs_dim() const override;
    const BetConfig& bet_config() const override { return bet_cfg_; }

    StepResult reset() override;
    StepResult step(int action) override;

    int current_player() const override;
    torch::Tensor observation() const override;
    torch::Tensor legal_action_mask() const override;
    bool is_terminal() const override;

    // ── public state accessors (used by the interactive play CLI) ──────
    // All values are in mbb (millibigblinds) where applicable, matching
    // the game engine's internal units.
    std::array<int, 2> hole_cards(int player) const;
    int community_count() const;
    int community_card(int idx) const;
    int pot() const;
    int current_bet() const;
    int stack(int player) const;
    int round() const;
    int raise_num() const;
    /// Terminal utility for `player` in mbb. Undefined if !is_terminal().
    int terminal_utility(int player) const;
    const ::Game::DefaultGameConfig& game_config() const { return poker_cfg_.game; }

private:
    void auto_advance_chance();
    void rebuild_action_table();
    torch::Tensor compute_mask() const;

    PokerConfig         poker_cfg_;
    BetConfig           bet_cfg_;
    std::mt19937        rng_;
    ::Game::DefaultBettingConfig          game_betting_cfg_;
    std::unique_ptr<::Game::DiscreteGame> game_;

    // Obs construction is delegated to ObservationBuilder. The env owns
    // the bet-history buffer + the layout/normalisers; the builder owns
    // the per-tensor write logic.
    ObservationBuilder obs_builder_;

    int   A_              = 0;
    int   allin_slot_     = -1;     // -1 if disabled
    float reward_norm_    = 1.0f;   // = 10 * big_blind; terminal-reward scaling

    // PPO-action-index → Poker Action, or nullopt if illegal this state.
    std::vector<std::optional<::Game::Action>> action_table_;

    // Hand-level bet history (cleared on reset, appended in step).
    std::vector<BetHistoryEntry> bet_history_;
};

class PokerEnvironmentFactory : public IPokerEnvironmentFactory {
public:
    explicit PokerEnvironmentFactory(PokerConfig poker_cfg);

    std::unique_ptr<IPokerEnvironment> create(const BetConfig& cfg) override;

private:
    PokerConfig poker_cfg_;
    std::atomic<uint64_t> instance_counter_{0};
};

} // namespace poker_ppo
