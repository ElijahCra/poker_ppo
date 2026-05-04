#pragma once

#include "config.h"          // PokerConfig + kPokerConfig live here
#include "environment.h"
#include "observation_builder.h"
#include "Game.hpp"

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace poker_ppo {

// PokerEnvironment — wraps Game::DiscreteGame (HU NLHE) in the
// IPokerEnvironment interface. Construction knobs come from PokerConfig
// (config.h). Auto-advances through ChanceState transitions, so PPO only
// sees ActionState or terminal snapshots.
//
// Action space (indices into the PPO policy's categorical):
//   0                       → Fold
//   1                       → Check or Call (whichever is legal)
//   2 .. 1+K                → K pot-fraction raises from PokerConfig
//   2+K  (if allow_all_in)  → All-in raise (only when distinct)
//
// Reward: the terminal reward is seat-0's utility delta in mbb,
// normalised by 10·big_blind. PPO handles the sign-flip for seat 1.

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
