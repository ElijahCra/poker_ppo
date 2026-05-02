#pragma once
//
// config.h — single source of truth for compile-time training/eval config.
//
// All hyperparameters that don't depend on runtime input live here as
// `inline constexpr` instances. Edit the values, rebuild, and the
// configuration is baked into the binary. Branches conditioned on these
// values via `if constexpr (config::kFoo.flag)` are dead-code-eliminated
// at compile time.
//
// Game-rule config (`Game::kGameConfig`, `Game::kBettingConfig`) and the
// env-side wrapper (`kPokerConfig`) are also `inline constexpr`, defined
// in `Game/GameConfig.hpp`, `Game/BettingConfig.hpp`, and `poker_env.h`
// respectively. They reference `kPPOConfig.hist` / `kPPOConfig.round_summary`
// here so the env observation layout cannot drift from the network input
// layout.
//

#include "types.h"
#include "best_response.h"
#include "features.h"

namespace poker_ppo::config {

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

}  // namespace poker_ppo::config
