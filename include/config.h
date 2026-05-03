#pragma once
//
// config.h — single source of truth for compile-time training/eval config.
//
// Holds both the *type* definitions for the trainer-side configs (PPO,
// bet-history attention, round-summary features, opponent pool) and the
// `inline constexpr` instances of those configs that the binary is
// compiled against. Edit values, rebuild, branches conditioned on
// `if constexpr (config::kFoo.flag)` are dead-code-eliminated.
//
// Game-rule config (`Game::kGameConfig`, `Game::kBettingConfig`) lives
// alongside its types in `Game/GameConfig.hpp` / `Game/BettingConfig.hpp`.
// The env-side `kPokerConfig` is in `poker_env.h` and references
// `kPPOConfig.hist` / `kPPOConfig.round_summary` here, so the env
// observation layout cannot drift from the network input layout.
//

#include "types.h"
#include "features.h"

#include <cstdint>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// Bet-history attention encoder config
// ─────────────────────────────────────────────────────────────────────────────
//
// The env exposes a variable-length sequence of past betting actions (this
// hand) as a fixed-size, padded block at the tail of the observation
// vector. The network slices that block back into a [T, F] token tensor
// and runs a small Transformer encoder over it. A learnable CLS token is
// prepended so attention has a well-defined output for empty histories.
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

    bool enabled         = true;  // master switch for the attention encoder
    int  max_history_len = 32;    // T — padded length of the action sequence
    int  attn_dim        = 64;    // D — token embedding & attention model dim
    int  attn_heads      = 4;     // H — multi-head attention heads (D % H == 0)
    int  ffn_mult        = 4;     // FFN inner dim = ffn_mult * attn_dim
    int  num_blocks      = 1;     // stacked attention blocks

    /// Total trailing block in the observation vector:
    ///   T (mask) + T * F (token features)   = T * (1 + F)
    /// Returns 0 unconditionally when the build-time flag
    /// `features::ATTENTION_ENCODER` is false, so the entire history block
    /// is stripped from the obs layout in that build.
    [[nodiscard]] constexpr int history_block_dim() const noexcept {
        if constexpr (!features::ATTENTION_ENCODER) return 0;
        return enabled ? max_history_len * (1 + feat_per_action) : 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Round-summary feature block
// ─────────────────────────────────────────────────────────────────────────────
//
// A hand-engineered, fixed-size alternative (or complement) to the
// attention encoder. For each of the four betting rounds, the env emits
// four features computed from the current hand's bet history:
//   0  my_chips_in     — chip delta I put in this round  / initial_stack
//   1  opp_chips_in    — chip delta opponent put in      / initial_stack
//   2  raises_count    — total raises this round         / max_raises_per_round
//   3  i_am_aggressor  — 1 if I was the last to raise this round, else 0
//
// All features are computed from the acting player's perspective, so the
// shared network can be used for both seats. Block size = 4 rounds × 4 =
// 16 floats (when enabled), appended to the obs after the static features.
//
struct RoundSummaryConfig {
    static constexpr int feat_per_round = 4;
    static constexpr int num_rounds     = 4;

    bool enabled = false;

    /// Block size in floats. Returns 0 unconditionally when the build-
    /// time flag `features::ROUND_SUMMARY` is false.
    [[nodiscard]] constexpr int dim() const noexcept {
        if constexpr (!features::ROUND_SUMMARY) return 0;
        return enabled ? num_rounds * feat_per_round : 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Opponent pool — frozen past-policy snapshots for self-play stabilisation
// ─────────────────────────────────────────────────────────────────────────────
//
// Empty / disabled by default → full self-play. When enabled, every
// `snapshot_every` updates the trainer copies the live network into a
// FIFO pool of size `max_size`. During rollouts, each env independently
// draws an opponent: with probability `p_use_pool` it pulls a pool member
// to play the non-learner seat; otherwise it stays in current-vs-current
// self-play. Snapshots are only sampled after `warmup_updates` so the
// pool isn't filled with cold-start garbage.
//
struct OpponentPoolConfig {
    bool     enabled                = false;
    int      max_size               = 20;
    int      snapshot_every         = 200;
    int      warmup_updates         = 200;
    float    p_use_pool             = 0.5f;
    // Cap on distinct pool snapshots used in a single rollout. Lower =
    // cheaper inference (fewer batched forwards) at the cost of less
    // opponent diversity within a rollout — diversity recovers across
    // rollouts as we resample at each rollout start.
    int      max_unique_per_rollout = 1;
    uint64_t seed                   = 0;   // 0 → random_device
};

// ─────────────────────────────────────────────────────────────────────────────
// PPO hyper-parameters
// ─────────────────────────────────────────────────────────────────────────────

struct PPOConfig {
    // ── core PPO ────────────────────────────────────────────────────────
    float gamma            = 1.0f;
    float gae_lambda       = 1.0f;
    float clip_coef        = 0.1f;
    float ent_coef         = 0.01f;
    float vf_coef          = 0.5f;
    float max_grad_norm    = 0.5f;
    bool  clip_vloss       = true;
    bool  norm_advantages  = true;

    // ── optimiser ───────────────────────────────────────────────────────
    float learning_rate    = 2.5e-4f;
    bool  anneal_lr        = true;
    // Floor on the annealed LR as a fraction of `learning_rate`. Default
    // linear-to-zero schedule does negligible learning in the last ~quarter
    // of training; floor at 0.1× to keep updates meaningful all the way
    // through. Set to 0 to recover linear-to-zero.
    float min_lr_frac      = 0.1f;

    // Entropy-coefficient cosine schedule. When enabled, `ent_coef` is the
    // start value and the effective coefficient decays to `ent_coef_min`
    // over the full run via:
    //     ent(t) = ent_min + 0.5·(ent_start − ent_min)·(1 + cos(π·t/T))
    // Rationale (poker IIGs): optimal entropy is high early (action-space
    // exploration) and low late (sharpen near-deterministic regions:
    // premium-hand value-betting, trash folds). A constant ent_coef can't
    // optimise both.
    bool  anneal_ent_coef  = false;
    float ent_coef_min     = 0.01f;

    // ── rollout ─────────────────────────────────────────────────────────
    int   num_envs         = 8;
    int   num_steps        = 128;
    int   update_epochs    = 4;
    int   num_minibatches  = 4;

    // ── training ────────────────────────────────────────────────────────
    int   total_timesteps  = 10'000'000;

    // ── network ─────────────────────────────────────────────────────────
    int   hidden_dim       = 512;
    int   num_layers       = 3;
    BetHistoryConfig    hist;
    RoundSummaryConfig  round_summary;

    // ── opponent pool ───────────────────────────────────────────────────
    OpponentPoolConfig  opp_pool;

    // ── derived ─────────────────────────────────────────────────────────
    constexpr int batch_size()     const noexcept { return num_envs * num_steps; }
    constexpr int minibatch_size() const noexcept { return batch_size() / num_minibatches; }
    constexpr int num_updates()    const noexcept { return total_timesteps / batch_size(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Best-response (approximate exploitability) evaluator config
// ─────────────────────────────────────────────────────────────────────────────
//
// See best_response.h for the algorithm; this struct is co-located here so
// `kBRConfig` (below) can be `inline constexpr` without pulling the
// evaluator's heavy implementation into config.h.
//
struct BestResponseConfig {
    bool  enabled            = false;
    int   eval_every         = 1000;   // run every N main-trainer updates
    int   updates_per_eval   = 200;    // PPO updates of exploiter per eval

    int   num_envs           = 32;
    int   num_steps          = 128;
    int   update_epochs      = 4;
    int   num_minibatches    = 4;

    float learning_rate      = 3.0e-4f;
    float ent_coef           = 0.01f;
    float vf_coef            = 0.5f;
    float clip_coef          = 0.2f;
    float max_grad_norm      = 0.5f;
    float gamma              = 1.0f;
    float gae_lambda         = 1.0f;
    bool  norm_advantages    = true;
    bool  clip_vloss         = false;

    bool  warm_start         = true;

    // Fresh exploiter seeds per evaluate() call. >1 trains independent
    // exploiters from random init against the same frozen target and
    // reports max-bb/hand as the canonical lower bound (mean/min/std are
    // diagnostics). Each seed's bb/hand is a valid lower bound on its own;
    // max over seeds is the tightest bound the budget can produce. Cost
    // scales linearly. With >1, warm_start is ignored.
    int   num_exploiter_seeds = 3;

    // Hands played in the post-training eval-only match between the
    // trained exploiter and frozen target. The match's bb/hand is the
    // reported BR estimate — using rewards collected during the exploiter's
    // training rollouts would bias the bound downward (early-training
    // exploiter plays badly while learning). Set to 0 to skip and fall
    // back to training-time reward averaging (debug only).
    int   eval_hands         = 5000;

    float bb_per_unit_reward = 10.0f;  // matches PokerEnvironment::reward_norm
    uint64_t seed            = 0;
};

namespace config {

// ─── Bet abstraction ────────────────────────────────────────────────────
// Discrete-action layout exposed to PPO. Must match the chosen game
// variant's GameConfig (validated at PokerEnvironment construction).
inline constexpr BetConfig kBetConfig{
    .num_raise_sizes    = 12,    // 11 pot fractions + AI; matches main.cpp
    .min_raise          = 0.5,
    .geometric_ratio    = 1.5,
    .max_bets_per_round = 4,
};

// ─── PPO trainer ────────────────────────────────────────────────────────
// Hyperparameters captured from the most recent main.cpp configuration.
// Tweak here, rebuild, and the trainer picks them up. Code uses
// `if constexpr (kPPOConfig.flag)` to strip dead branches.
inline constexpr PPOConfig kPPOConfig{
    // Core PPO
    .gamma            = 1.0f,
    .gae_lambda       = 1.0f,
    .clip_coef        = 0.1f,
    .ent_coef         = 0.06f,    // higher than typical RL — see Rudolph et al. 2026
    .vf_coef          = 0.5f,
    .max_grad_norm    = 0.5f,
    .clip_vloss       = true,
    .norm_advantages  = true,

    // Optimiser
    .learning_rate    = 3.0e-4f,
    .anneal_lr        = true,
    .min_lr_frac      = 0.1f,

    // Entropy schedule (cosine to ent_coef_min)
    .anneal_ent_coef  = true,
    .ent_coef_min     = 0.01f,

    // Rollout
    .num_envs         = 96,
    .num_steps        = 128,
    .update_epochs    = 4,
    .num_minibatches  = 4,

    // Training
    .total_timesteps  = 300'000'000,

    // Network
    .hidden_dim       = 256,
    .num_layers       = 2,
    .hist             = BetHistoryConfig{
        .enabled         = false,    // build-time gate is features::ATTENTION_ENCODER
        .max_history_len = 32,
        .attn_dim        = 64,
        .attn_heads      = 4,
        .ffn_mult        = 4,
        .num_blocks      = 1,
    },
    .round_summary    = RoundSummaryConfig{
        .enabled = false,            // build-time gate is features::ROUND_SUMMARY
    },

    // Opponent pool (reservoir-sampled past selves)
    .opp_pool         = OpponentPoolConfig{
        .enabled                 = true,
        .max_size                = 20,
        .snapshot_every          = 200,
        .warmup_updates          = 400,
        .p_use_pool              = 0.05f,
        .max_unique_per_rollout  = 4,
        .seed                    = 0,
    },
};

// ─── Best-response evaluator ────────────────────────────────────────────
// Used by main.cpp to instrument approximate exploitability during
// training. All knobs are compile-time. Set `enabled = false` to remove
// BR overhead from the binary.
inline constexpr BestResponseConfig kBRConfig{
    .enabled            = true,
    .eval_every         = 1000,
    .updates_per_eval   = 1000,
    .num_envs           = 32,
    .num_steps          = 128,
    .update_epochs      = 4,
    .num_minibatches    = 4,
    .learning_rate      = 3.0e-4f,
    .ent_coef           = 0.01f,
    .vf_coef            = 0.5f,
    .clip_coef          = 0.2f,
    .max_grad_norm      = 0.5f,
    .gamma              = 1.0f,
    .gae_lambda         = 1.0f,
    .norm_advantages    = true,
    .clip_vloss         = false,
    .warm_start         = false,
    .num_exploiter_seeds = 3,
    .eval_hands         = 5000,
    .bb_per_unit_reward = 10.0f,
    .seed               = 0xCAFEBABEull,
};

}  // namespace config
}  // namespace poker_ppo
