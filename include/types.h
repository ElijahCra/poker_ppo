#pragma once

#include <torch/torch.h>
#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <optional>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// Betting configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Controls the discrete action space for betting.
///
/// Non-raise actions are always: {Fold, Check/Call}.
/// Raise actions use geometrically-spaced bet sizes:
///   bet_i = min_raise * ratio^i   for i in [0, num_raise_sizes)
///
/// Example with min_raise=1.0, ratio=2.0, num_raise_sizes=4:
///   raises = {1x pot, 2x pot, 4x pot, 8x pot}   (or whatever unit you choose)
///
struct BetConfig {
    int    num_raise_sizes  = 4;      // number of distinct raise amounts
    double min_raise        = 1.0;    // smallest raise (in your chosen unit)
    double geometric_ratio  = 2.0;    // multiplier between successive raises
    int    max_bets_per_round = 4;    // cap on total bet/raise actions per
                                      // player per betting round

    /// Total number of discrete actions = fold + check/call + raises.
    int action_count() const { return 2 + num_raise_sizes; }

    /// Returns the actual raise amount for raise index i ∈ [0, num_raise_sizes).
    double raise_amount(int i) const {
        assert(i >= 0 && i < num_raise_sizes);
        return min_raise * std::pow(geometric_ratio, i);
    }

    /// Convenience: all raise amounts as a vector.
    std::vector<double> all_raise_amounts() const {
        std::vector<double> v(num_raise_sizes);
        for (int i = 0; i < num_raise_sizes; ++i)
            v[i] = raise_amount(i);
        return v;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Action encoding
// ─────────────────────────────────────────────────────────────────────────────

/// Action indices:
///   0         → Fold
///   1         → Check / Call
///   2 .. 2+N  → Raise(raise_amount(i - 2))
///
namespace Action {
    constexpr int Fold      = 0;
    constexpr int CheckCall = 1;

    /// Convert a raise index (0-based) into the flat action id.
    inline int Raise(int raise_idx) { return 2 + raise_idx; }

    /// True if action id represents a raise.
    inline bool is_raise(int action_id) { return action_id >= 2; }

    /// Extract the raise index from a flat action id.
    inline int raise_index(int action_id) {
        assert(is_raise(action_id));
        return action_id - 2;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Observation / step data
// ─────────────────────────────────────────────────────────────────────────────

/// What the environment returns after a step (or reset).
struct StepResult {
    torch::Tensor observation;        // [obs_dim]  float
    float         reward  = 0.0f;     // reward for the acting player
    bool          done    = false;    // episode over?
    torch::Tensor legal_action_mask;  // [action_count]  bool / float mask
                                      // 1 = legal, 0 = illegal
};

// ─────────────────────────────────────────────────────────────────────────────
// PPO hyper-parameters
// ─────────────────────────────────────────────────────────────────────────────

struct PPOConfig {
    // ── core PPO ────────────────────────────────────────────────────────
    float  gamma            = 0.99f;
    float  gae_lambda       = 0.95f;
    float  clip_coef        = 0.1f;
    float  ent_coef         = 0.05f;   // high entropy helps in IIGs (see paper)
    float  vf_coef          = 0.5f;
    float  max_grad_norm    = 0.5f;
    bool   clip_vloss       = true;
    bool   norm_advantages  = true;

    // Per-step reward multiplier applied before the reward enters the buffer.
    // For NLHE with 100k-mbb stacks, set to 1/initial_stack so returns live in
    // [-1, 1] — keeps the critic's target magnitude sane and prevents vf_coef
    // from dominating the shared-trunk gradient. Elo / league reporting still
    // reads the raw env reward, so display semantics are unchanged.
    float  reward_scale     = 1.0f;

    // ── optimiser ───────────────────────────────────────────────────────
    float  learning_rate    = 2.5e-4f;
    bool   anneal_lr        = true;

    // ── rollout ─────────────────────────────────────────────────────────
    int    num_envs         = 8;       // parallel self-play games
    int    num_steps        = 128;     // steps per env per rollout
    int    update_epochs    = 4;       // passes over the rollout buffer
    int    num_minibatches  = 4;

    // ── training ────────────────────────────────────────────────────────
    int    total_timesteps  = 10'000'000;

    // ── network ─────────────────────────────────────────────────────────
    int    hidden_dim       = 512;
    int    num_layers       = 3;

    // ── derived ─────────────────────────────────────────────────────────
    int batch_size()     const { return num_envs * num_steps; }
    int minibatch_size() const { return batch_size() / num_minibatches; }
    int num_updates()    const { return total_timesteps / batch_size(); }
};

} // namespace poker_ppo
