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

// Wraps a Game::GameConfig (the poker variant) plus PPO-side knobs.
// All poker-rule tuning lives in `game`; only per-run seeding and adapter
// flags belong on PokerConfig itself.
struct PokerConfig {
    ::Game::GameConfig  game{};           // deck, stacks, blinds, betting structure
    BetHistoryConfig    hist{};           // attention-encoder layout (T, F, ...)
    RoundSummaryConfig  round_summary{};  // per-round summary feature block

    // Seed used as a base — each created env gets seed ^ instance hash.
    uint64_t seed = 0x9E3779B97F4A7C15ull;

    [[nodiscard]] int num_raise_slots() const { return game.num_raise_slots(); }
    [[nodiscard]] int action_count()    const { return game.action_count(); }
};

class PokerEnvironment : public IPokerEnvironment {
public:
    PokerEnvironment(const PokerConfig& poker_cfg,
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
    const ::Game::GameConfig& game_config() const { return poker_cfg_.game; }

private:
    // Per-action snapshot recorded each time the env applies a player action.
    // Hand-level: cleared by reset(). Truncated to the most-recent
    // `hist_cfg_.max_history_len` entries when serialised.
    struct HistEntry {
        uint32_t amount;        // chip delta added to pot (mbb)
        uint8_t  player;        // 0 or 1 (seat that took the action)
        uint8_t  round;         // 0..3
        bool     is_aggressive; // true = Raise, false = Call/Check/Fold
    };

    void auto_advance_chance();
    void rebuild_action_table();
    torch::Tensor compute_observation() const;
    torch::Tensor compute_mask() const;

    /// Serialise `bet_history_` into a [T, F] tokens block + [T] mask, written
    /// into `obs` at offset `dst_off`.  Layout: mask first, then tokens (row-
    /// major).  Returns the new offset (= dst_off + T*(1+F)).
    int write_history_block(torch::TensorAccessor<float, 1>& a, int dst_off,
                            int current_player) const;

    /// Write the per-round summary block (4 rounds × 4 features) at offset
    /// `dst_off` and return the new offset.  Derived live from `bet_history_`
    /// so it stays consistent with the attention encoder when both are on.
    int write_round_summary_block(torch::TensorAccessor<float, 1>& a,
                                  int dst_off, int current_player) const;

    PokerConfig         poker_cfg_;
    BetConfig           bet_cfg_;
    BetHistoryConfig    hist_cfg_;
    RoundSummaryConfig  rs_cfg_;
    std::mt19937 rng_;
    ::Game::BettingConfig game_betting_cfg_;
    std::unique_ptr<::Game::DiscreteGame> game_;

    int obs_dim_       = 0;
    int static_obs_dim_ = 0;     // obs_dim_ minus the history tail
    int A_             = 0;
    int allin_slot_    = -1;     // -1 if disabled
    float stack_norm_  = 1.0f;   // = game.initial_stack; observation scaling
    float pot_norm_    = 1.0f;   // = 2 * initial_stack
    float reward_norm_ = 1.0f;   // = 10 * big_blind; terminal-reward scaling
    int   max_raises_norm_ = 4;  // = game.max_raises_per_round (>=1 guarded)

    // PPO-action-index → Poker Action, or nullopt if illegal this state.
    std::vector<std::optional<::Game::Action>> action_table_;

    // Hand-level bet history (cleared on reset, appended in step).
    std::vector<HistEntry> bet_history_;
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
