#pragma once

#include "environment.h"
#include "Game.hpp"
#include "GameConfig.hpp"

#include <atomic>
#include <memory>
#include <random>
#include <vector>
#include <optional>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// QFR environment adapter
// ─────────────────────────────────────────────────────────────────────────────
//
// Wraps Game::DiscreteGame (heads-up no-limit hold'em from the QFR project) in
// the IPokerEnvironment interface used by PPO.
//
// Action space (indices into the PPO policy's categorical):
//   0                       → Fold
//   1                       → Check or Call (whichever is legal)
//   2 .. 1+K                → K pot-fraction raises defined in QFRConfig
//   2+K  (if allow_all_in)  → All-in raise (only exposed when distinct)
//
// The adapter auto-advances through ChanceState transitions (which in the
// QFR implementation just flip the round marker; all cards are dealt at hand
// start).  PPO therefore only ever sees ActionState or terminal snapshots.
//
// Reward: the terminal reward is the utility delta for seat 0 (in "mbb" /
// millibigblinds as stored by QFR), normalised by INITIAL_STACK so it sits in
// a reasonable range for the critic.  PPO handles the sign-flip for seat 1.

// Wraps a Game::GameConfig (the poker variant) plus PPO-side knobs.
// All poker-rule tuning lives in `game`; only per-run seeding and adapter
// flags belong on QFRConfig itself.
struct QFRConfig {
    ::Game::GameConfig game{};    // deck, stacks, blinds, betting structure

    // Seed used as a base — each created env gets seed ^ instance hash.
    uint64_t seed = 0x9E3779B97F4A7C15ull;

    [[nodiscard]] int num_raise_slots() const { return game.num_raise_slots(); }
    [[nodiscard]] int action_count()    const { return game.action_count(); }
};

class QFRPokerEnvironment : public IPokerEnvironment {
public:
    QFRPokerEnvironment(const QFRConfig& qfr_cfg,
                        const BetConfig& bet_cfg,
                        uint64_t seed);

    int obs_dim() const override { return obs_dim_; }
    const BetConfig& bet_config() const override { return bet_cfg_; }

    StepResult reset() override;
    StepResult step(int action) override;

    int current_player() const override;
    torch::Tensor observation() const override;
    torch::Tensor legal_action_mask() const override;
    bool is_terminal() const override;

private:
    void auto_advance_chance();
    void rebuild_action_table();
    torch::Tensor compute_observation() const;
    torch::Tensor compute_mask() const;

    QFRConfig qfr_cfg_;
    BetConfig bet_cfg_;
    std::mt19937 rng_;
    ::Game::BettingConfig game_betting_cfg_;
    std::unique_ptr<::Game::DiscreteGame> game_;

    int obs_dim_ = 0;
    int A_       = 0;
    int allin_slot_ = -1;        // -1 if disabled
    float stack_norm_ = 1.0f;    // = game.initial_stack; observation scaling
    float pot_norm_   = 1.0f;    // = 2 * initial_stack
    int   max_raises_norm_ = 4;  // = game.max_raises_per_round (>=1 guarded)

    // PPO-action-index → QFR Action, or nullopt if illegal this state.
    std::vector<std::optional<::Game::Action>> action_table_;
};

class QFRPokerEnvironmentFactory : public IPokerEnvironmentFactory {
public:
    explicit QFRPokerEnvironmentFactory(QFRConfig qfr_cfg);

    std::unique_ptr<IPokerEnvironment> create(const BetConfig& cfg) override;

private:
    QFRConfig qfr_cfg_;
    std::atomic<uint64_t> instance_counter_{0};
};

} // namespace poker_ppo
