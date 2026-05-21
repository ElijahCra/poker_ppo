#pragma once
// Compile-time training/eval config.
// Game-rule config lives in Game/GameConfig.hpp.

#include "types.h"
#include "base_configs.h"

namespace poker_ppo{
namespace config {

// Discrete action layout
static constexpr BetConfig kBetConfig{
    .num_raise_sizes    = 12,
    .min_raise          = 0.5,
    .geometric_ratio    = 1.5,
    .max_bets_per_round = 4,
};

//PPO hyperparameters + settings
static constexpr PPOConfig kPPOConfig{
    .gamma            = 1.0f,
    .gae_lambda       = 0.90f,
    .clip_coef        = 0.1f,
    .ent_coef         = 0.4f,
    .vf_coef          = 0.5f,
    .max_grad_norm    = 0.5f,
    .clip_vloss       = true,
    .norm_advantages  = true,

    .learning_rate    = 3.0e-4f,
    .anneal_lr        = true,
    .min_lr_frac      = 0.2f,

    .anneal_ent_coef  = true,
    .ent_coef_min     = 0.01f,

    .num_envs         = 96,
    .num_steps        = 128,
    .update_epochs    = 4,
    .num_minibatches  = 4,

    .total_timesteps  = 600'000'000,

    .hidden_dim       = 512,
    .num_layers       = 4,
    .hist             = BetHistoryConfig{
        .enabled         = true,    // build gate: features::ATTENTION_ENCODER
        .max_history_len = 16,       // T² attn cost
        .attn_dim        = 96,
        .attn_heads      = 4,
        .ffn_mult        = 3,        // FF hidden = 128, half the trunk width
        .num_blocks      = 2,
    },
    .round_summary    = RoundSummaryConfig{
        .enabled = true,            // build gate: features::ROUND_SUMMARY
    },

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

static constexpr BestResponseConfig kBRConfig{
    .enabled            = true,
    .eval_every         = 3000,
    .updates_per_eval   = 3000,
    .num_envs           = 32,
    .num_steps          = 128,
    .update_epochs      = 4,
    .num_minibatches    = 4,
    .learning_rate      = 1.5e-4f,
    .ent_coef           = 0.03f,
    .vf_coef            = 0.5f,
    .clip_coef          = 0.2f,
    .max_grad_norm      = 0.5f,
    .gamma              = 1.0f,
    .gae_lambda         = 1.0f,
    .norm_advantages    = true,
    .clip_vloss         = false,
    .warm_start         = true,
    .num_exploiter_seeds = 1,
    .eval_hands         = 10000,
    .bb_per_unit_reward = 10.0f,
    .seed               = 0xCAFEBABEull,
};

}  // namespace config

// Pinned to kPPOConfig so env obs layout can't drift from network input.
static constexpr PokerConfig kPokerConfig{
    .game          = Game::kGameConfig,
    .hist          = config::kPPOConfig.hist,
    .round_summary = config::kPPOConfig.round_summary,
    .seed          = 0x12345ULL,
};

// Compile time checks mirroring the runtime check in PokerEnvironment's constructor
static_assert(kPokerConfig.action_count() == config::kBetConfig.action_count(),
              "kBetConfig.action_count() must match kPokerConfig.action_count() "
              "(2 + num_raise_slots)");
static_assert(kPokerConfig.game.max_raises_per_round
              == static_cast<uint8_t>(config::kBetConfig.max_bets_per_round),
              "BetConfig.max_bets_per_round must match game.max_raises_per_round");

}  // namespace poker_ppo
