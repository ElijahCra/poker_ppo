#pragma once

#include "types.h"
#include <torch/torch.h>
#include <memory>
#include <vector>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// IPokerEnvironment  — abstract interface you implement
// ─────────────────────────────────────────────────────────────────────────────
//
// Contract:
//
//  • The environment manages a *single* Texas Hold'em hand at a time.
//    PPO will create `num_envs` independent instances for parallel rollout.
//
//  • Two-player, zero-sum, self-play.  The PPO agent controls *both* seats.
//    The environment is responsible for tracking whose turn it is.
//
//  • `observation()` returns a feature vector visible to the *acting* player.
//    You decide what goes in it (cards, pot, stack, history encoding, …).
//
//  • `legal_action_mask()` returns a float tensor of shape [action_count]
//    where 1.0 = legal and 0.0 = illegal.  The PPO agent will mask logits
//    before sampling.  You must respect BetConfig semantics:
//      index 0         → fold   (always legal unless check is free)
//      index 1         → check / call
//      index 2 .. 2+N  → raise by bet_config.raise_amount(i - 2)
//    If the player has already hit max_bets_per_round, mask all raises to 0.
//
//  • `step()` advances the game by one *player action*.  If the action ends
//    a betting round, you should deal community cards internally before the
//    next call to `observation()`.  When the hand is over, set done=true and
//    return the reward *from player 1's perspective* (player 2 gets -reward).
//
//  • `current_player()` returns 0 or 1 (seat index of acting player).
//
// ─────────────────────────────────────────────────────────────────────────────

class IPokerEnvironment {
public:
    virtual ~IPokerEnvironment() = default;

    // ── configuration ───────────────────────────────────────────────────

    /// Dimensionality of the observation vector.
    virtual int obs_dim() const = 0;

    /// The bet configuration this environment was constructed with.
    virtual const BetConfig& bet_config() const = 0;

    // ── episode lifecycle ───────────────────────────────────────────────

    /// Reset the environment to a fresh hand.  Shuffle and deal hole cards.
    /// Returns the initial StepResult (reward=0, done=false).
    virtual StepResult reset() = 0;

    /// Apply an action (index into the discrete action space).
    /// Returns the resulting observation, reward, done flag, and new mask.
    /// Reward is from player 1's perspective.
    virtual StepResult step(int action) = 0;

    // ── queries ─────────────────────────────────────────────────────────

    /// Which seat (0 or 1) is currently acting?
    virtual int current_player() const = 0;

    /// Current observation tensor for the acting player.  Shape: [obs_dim].
    virtual torch::Tensor observation() const = 0;

    /// Legal-action mask for the acting player.  Shape: [action_count].
    virtual torch::Tensor legal_action_mask() const = 0;

    /// Is the current hand finished?
    virtual bool is_terminal() const = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// IPokerEnvironmentFactory  — creates environment instances
// ─────────────────────────────────────────────────────────────────────────────

class IPokerEnvironmentFactory {
public:
    virtual ~IPokerEnvironmentFactory() = default;

    /// Create a new environment instance with the given bet configuration.
    virtual std::unique_ptr<IPokerEnvironment> create(const BetConfig& cfg) = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// VectorizedEnv  — thin wrapper that owns N independent environments
// ─────────────────────────────────────────────────────────────────────────────
//
// This is provided for you; no need to implement it.  PPO uses it internally
// to run `num_envs` games in parallel (on the same thread for simplicity —
// these games are tiny).

class VectorizedEnv {
public:
    VectorizedEnv(IPokerEnvironmentFactory& factory,
                  const BetConfig& cfg,
                  int num_envs);

    int  num_envs()     const { return static_cast<int>(envs_.size()); }
    int  obs_dim()      const { return envs_[0]->obs_dim(); }
    int  action_count() const { return envs_[0]->bet_config().action_count(); }

    /// Reset all environments.  Returns stacked observations [num_envs, obs_dim].
    torch::Tensor reset_all();

    /// Step each environment with its corresponding action.
    /// actions: vector of int, length num_envs.
    /// Returns: observations [N, obs_dim], rewards [N], dones [N], masks [N, A].
    struct BatchStepResult {
        torch::Tensor observations;      // [N, obs_dim]
        torch::Tensor rewards;           // [N]
        torch::Tensor dones;             // [N]  float (0/1)
        torch::Tensor legal_action_masks; // [N, action_count]
        torch::Tensor current_players;   // [N]  int (0 or 1)
    };
    BatchStepResult step(const std::vector<int>& actions);

    /// Access individual envs (e.g. for current_player queries).
    IPokerEnvironment& env(int i) { return *envs_[i]; }

    /// Raw access to the owning vector of envs (for the coroutine rollout
    /// scheduler, which needs to drive each env independently).
    std::vector<std::unique_ptr<IPokerEnvironment>>& envs_mut() { return envs_; }

private:
    std::vector<std::unique_ptr<IPokerEnvironment>> envs_;

    // Pre-allocated CPU output tensors — reused every step to avoid heap allocs
    torch::Tensor obs_buf_;      // [N, obs_dim]
    torch::Tensor rewards_buf_;  // [N]
    torch::Tensor dones_buf_;    // [N]
    torch::Tensor masks_buf_;    // [N, A]
    torch::Tensor players_buf_;  // [N]  int32
};

} // namespace poker_ppo
