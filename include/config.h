#pragma once
// Compile-time training/eval config. Edit values, rebuild.
// Game-rule config lives in Game/GameConfig.hpp.

#include "types.h"
#include "features.h"
#include "GameConfig.hpp"

#include <cstdint>
#include <random>

namespace poker_ppo {

// Bet-history attention encoder. Per-action features (F=8):
//   0  amount / initial_stack
//   1  amount / (2 * initial_stack)
//   2  is_my_action
//   3  is_aggressive
//   4..7  round one-hot
struct BetHistoryConfig {
    static constexpr int feat_per_action = 8;

    bool enabled         = true;
    int  max_history_len = 32;    // T
    int  attn_dim        = 64;    // D, must be divisible by attn_heads
    int  attn_heads      = 4;
    int  ffn_mult        = 4;
    int  num_blocks      = 1;

    // T mask + T*F tokens. Returns 0 when features::ATTENTION_ENCODER is off.
    [[nodiscard]] constexpr int history_block_dim() const noexcept {
        if constexpr (!features::ATTENTION_ENCODER) return 0;
        return enabled ? max_history_len * (1 + feat_per_action) : 0;
    }
};

// Per-round summary block. 4 features × 4 rounds = 16 floats.
//   0  my_chips_in / initial_stack
//   1  opp_chips_in / initial_stack
//   2  raises_count / max_raises_per_round
//   3  i_am_aggressor
struct RoundSummaryConfig {
    static constexpr int feat_per_round = 4;
    static constexpr int num_rounds     = 4;

    bool enabled = false;

    [[nodiscard]] constexpr int dim() const noexcept {
        if constexpr (!features::ROUND_SUMMARY) return 0;
        return enabled ? num_rounds * feat_per_round : 0;
    }
};

// Auxiliary counterfactual-value head: predicts payoff for every possible
// hole-card combo of the acting player at the current public state. Trains
// against showdown targets computed from observed self-play (both seats'
// cards visible to the trainer). Goal is range-aware trunk representations
// for downstream subgame-solving — see DeepStack/ReBeL.
//
// Output dim is C(52, 2) = 1326. Loss is masked: combos that share cards
// with the opponent's actual hand or the actual board are zeroed out.
struct CFVAuxConfig {
    bool  enabled = false;
    float coef    = 0.5f;   // weight in the total loss
};

inline constexpr int kCFVHeadDim = 1326;

// Reservoir-sampled past-policy snapshots for self-play stabilisation.
struct OpponentPoolConfig {
    bool     enabled                = false;
    int      max_size               = 20;
    int      snapshot_every         = 200;
    int      warmup_updates         = 200;
    float    p_use_pool             = 0.5f;
    // Cap on distinct snapshots used per rollout — bounds inference cost.
    int      max_unique_per_rollout = 1;
    uint64_t seed                   = 0;   // 0 → random_device
};

struct PPOConfig {
    float gamma            = 1.0f;
    float gae_lambda       = 1.0f;
    float clip_coef        = 0.1f;
    float ent_coef         = 0.01f;
    float vf_coef          = 0.5f;
    float max_grad_norm    = 0.5f;
    bool  clip_vloss       = true;
    bool  norm_advantages  = true;

    float learning_rate    = 2.5e-4f;
    bool  anneal_lr        = true;
    // Floor for the LR schedule, as a fraction of learning_rate. 0 = linear-to-zero.
    float min_lr_frac      = 0.1f;

    // Cosine entropy decay from ent_coef → ent_coef_min.
    // High entropy early helps exploration; low entropy late sharpens
    // near-deterministic regions. Constant ent_coef can't do both.
    bool  anneal_ent_coef  = false;
    float ent_coef_min     = 0.01f;

    int   num_envs         = 8;
    int   num_steps        = 128;
    int   update_epochs    = 4;
    int   num_minibatches  = 4;

    int   total_timesteps  = 10'000'000;

    int   hidden_dim       = 512;
    int   num_layers       = 3;
    BetHistoryConfig    hist;
    RoundSummaryConfig  round_summary;

    OpponentPoolConfig  opp_pool;
    CFVAuxConfig        cfv_aux;

    // ── MMD regularisation (Sokota et al., ICLR 2023) ────────────────────
    // When `kl_coef > 0`, adds `kl_coef * KL(π_θ || ρ)` to the PPO loss,
    // where ρ is a slowly-updated "magnet" snapshot of π_θ. This converts
    // vanilla self-play PPO into Magnetic Mirror Descent — a regularised
    // PG method with last-iterate Nash convergence guarantees in the
    // tabular case (and empirically in the deep variant). Cheap to add:
    // one extra forward pass per minibatch through the magnet network +
    // a clone of the policy every `magnet_update_every` updates.
    //
    // 0.05 is the value Sokota 2023 found best-on-average; the sweep
    // range was [2^-3..2^3] × that. Set kl_coef = 0 to disable (recovers
    // vanilla self-play PPO bit-for-bit).
    float kl_coef             = 0.05f;
    int   magnet_update_every = 100;  // updates between magnet refreshes

    constexpr int batch_size()     const noexcept { return num_envs * num_steps; }
    constexpr int minibatch_size() const noexcept { return batch_size() / num_minibatches; }
    constexpr int num_updates()    const noexcept { return total_timesteps / batch_size(); }
};

// Wraps Game::DefaultGameConfig + PPO-side knobs. `hist`/`round_summary`
// are pinned to PPOConfig at the kPokerConfig site so env obs layout
// stays in sync with network input layout.
struct PokerConfig {
    Game::DefaultGameConfig  game{};
    BetHistoryConfig    hist{};
    RoundSummaryConfig  round_summary{};

    // Base seed; each env gets seed ^ instance hash.
    uint64_t seed = std::random_device()();

    [[nodiscard]] constexpr int num_raise_slots() const noexcept { return game.num_raise_slots(); }
    [[nodiscard]] constexpr int action_count()    const noexcept { return game.action_count(); }
};

// Approximate best-response evaluator. See best_response.h.
struct BestResponseConfig {
    bool  enabled            = false;
    int   eval_every         = 1000;
    int   updates_per_eval   = 200;

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

    // >1 trains independent exploiters from random init and reports
    // max(bb/hand) — tightest lower bound the budget can produce.
    // With >1, warm_start is ignored.
    int   num_exploiter_seeds = 3;

    // Hands in the post-training eval match. Using training-time rewards
    // would bias the bound (early exploiter plays badly while learning).
    // 0 = skip and fall back to training rewards (debug only).
    int   eval_hands         = 5000;

    float bb_per_unit_reward = 10.0f;
    uint64_t seed            = std::random_device()();
};

namespace config {

// Discrete action layout. Must match the chosen game variant
// (validated at PokerEnvironment construction).
static constexpr BetConfig kBetConfig{
    .num_raise_sizes    = 12,
    .min_raise          = 0.5,
    .geometric_ratio    = 1.5,
    .max_bets_per_round = 4,
};

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
        .max_history_len = 16,       // HUNL caps actions at ~16/hand; T² attn cost
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
    .eval_every         = 3000,    // less frequent → cheaper overall, deeper per eval
    .updates_per_eval   = 3000,    // more chase time per eval
    .num_envs           = 32,
    .num_steps          = 128,
    .update_epochs      = 4,
    .num_minibatches    = 4,
    .learning_rate      = 1.5e-4f, // warm_start needs gentle LR; high LR shocks the exploiter
    .ent_coef           = 0.03f,   // more exploration to escape negative-BR plateau
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

// Mirrors the runtime check in PokerEnvironment's ctor; drift here fails
// the build instead of throwing at runtime.
static_assert(kPokerConfig.action_count() == config::kBetConfig.action_count(),
              "kBetConfig.action_count() must match kPokerConfig.action_count() "
              "(2 + num_raise_slots)");
static_assert(kPokerConfig.game.max_raises_per_round
              == static_cast<uint8_t>(config::kBetConfig.max_bets_per_round),
              "BetConfig.max_bets_per_round must match game.max_raises_per_round");

}  // namespace poker_ppo
