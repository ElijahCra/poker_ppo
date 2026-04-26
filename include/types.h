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
// Bet-history attention config
// ─────────────────────────────────────────────────────────────────────────────
//
// The environment exposes a *variable-length* sequence of past betting actions
// (this hand) as a fixed-size, padded block at the tail of the observation
// vector. The network slices that block back into a [T, F] token tensor and
// runs a small Transformer encoder over it.
//
// A learnable CLS token is prepended to the sequence so attention has a
// well-defined output even when the history is empty (preflop, before any
// action). The CLS output is concatenated with the static features.
//
// Per-action features (F = 8):
//   0  amount / initial_stack
//   1  amount / (2 * initial_stack)        (≈ pot-scale)
//   2  is_my_action  (1 if the *current* acting player made it)
//   3  is_aggressive (1 = Raise, 0 = Call/Check/Fold)
//   4  round one-hot[0]  (preflop)
//   5  round one-hot[1]  (flop)
//   6  round one-hot[2]  (turn)
//   7  round one-hot[3]  (river)
//
struct BetHistoryConfig {
    static constexpr int feat_per_action = 8;

    bool enabled         = true; // master switch for the attention encoder
    int  max_history_len = 32;   // T — padded length of the action sequence
    int  attn_dim        = 64;   // D — token embedding & attention model dim
    int  attn_heads      = 4;    // H — multi-head attention heads (D % H == 0)
    int  ffn_mult        = 4;    // FFN inner dim = ffn_mult * attn_dim
    int  num_blocks      = 1;    // stacked attention blocks

    /// Total trailing block in the observation vector:
    ///   T (mask) + T * F (token features)   = T * (1 + F)
    /// Returns 0 when the encoder is disabled — caller-side code that uses
    /// this to size obs / split tensors then falls back to a "static-only"
    /// layout automatically.
    [[nodiscard]] int history_block_dim() const {
        return enabled ? max_history_len * (1 + feat_per_action) : 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Round-summary feature block
// ─────────────────────────────────────────────────────────────────────────────
//
// A hand-engineered, fixed-size alternative (or complement) to the attention
// encoder.  For each of the four betting rounds (preflop, flop, turn, river)
// the env emits four features computed from the current hand's bet history:
//
//   0  my_chips_in     — chip delta I put in this round  / initial_stack
//   1  opp_chips_in    — chip delta opponent put in      / initial_stack
//   2  raises_count    — total raises this round         / max_raises_per_round
//   3  i_am_aggressor  — 1 if I was the last to raise this round, else 0
//
// All features are computed from the acting player's perspective so the same
// shared network can be used for both seats.  Block size = 4 rounds × 4 feats
// = 16 floats (when enabled) appended to the obs immediately after the static
// features and before the attention-history block.
//
struct RoundSummaryConfig {
    static constexpr int feat_per_round = 4;
    static constexpr int num_rounds     = 4;

    bool enabled = false;        // off by default; flip on with --round-summary

    [[nodiscard]] int dim() const {
        return enabled ? num_rounds * feat_per_round : 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// PPO hyper-parameters
// ─────────────────────────────────────────────────────────────────────────────

struct PPOConfig {
    // ── core PPO ────────────────────────────────────────────────────────
    float  gamma            = 0.9999f;
    float  gae_lambda       = 0.995f;
    float  clip_coef        = 0.1f;
    float  ent_coef         = 0.05f;   // high entropy helps in IIGs (see paper)
    float  vf_coef          = 0.5f;
    float  max_grad_norm    = 0.5f;
    bool   clip_vloss       = true;
    bool   norm_advantages  = true;

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
    BetHistoryConfig    hist;          // attention encoder over bet history
    RoundSummaryConfig  round_summary; // hand-engineered per-round features

    // ── derived ─────────────────────────────────────────────────────────
    int batch_size()     const { return num_envs * num_steps; }
    int minibatch_size() const { return batch_size() / num_minibatches; }
    int num_updates()    const { return total_timesteps / batch_size(); }
};

} // namespace poker_ppo
