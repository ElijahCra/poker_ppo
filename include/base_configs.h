#pragma once
#include <random>
#include "GameConfig.hpp"

// compile-time feature flags that control what's included in the binary
// disabled features result in no-op / shrunk model
namespace poker_ppo::features {
// Per-round normalised hand_indexer bucket features to approximate hand strength at every round
inline constexpr bool HAND_STRENGTH = true;

// Bet-history attention encoder: attention block over the per-action token sequence.
// the env writes a [T mask | T*F tokens] tail block in every observation
inline constexpr bool ATTENTION_ENCODER = true;

// Per-round summary block approximating betting aggression behavior
inline constexpr bool ROUND_SUMMARY = true;
}

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

// Reservoir-sampled past-policy snapshots for self-play stabilisation.
struct OpponentPoolConfig {
    bool     enabled                = false;
    int      max_size               = 20;
    int      snapshot_every         = 200;
    int      warmup_updates         = 200;
    float    p_use_pool             = 0.5f;
    // Cap on distinct snapshots used per rollout — bounds inference cost.
    int      max_unique_per_rollout = 1;
    uint64_t seed                   = 0;   // 0 -> random_device
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

    // Cosine entropy decay from ent_coef -> ent_coef_min.

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

    // Magnetic Mirror Descent (MMD) regularisation
    // When `kl_coef > 0`, adds `kl_coef * KL(π_θ || ρ)` to the PPO loss,
    // where ρ is a slowly-updated "magnet" snapshot of π_θ.

    float kl_coef             = 0.05f;
    int   magnet_update_every = 100;  // updates between magnet refreshes

    constexpr int batch_size()     const noexcept { return num_envs * num_steps; }
    constexpr int minibatch_size() const noexcept { return batch_size() / num_minibatches; }
    constexpr int num_updates()    const noexcept { return total_timesteps / batch_size(); }
};

// Wraps Game::DefaultGameConfig
struct PokerConfig {
    Game::DefaultGameConfig  game{};
    BetHistoryConfig    hist{};
    RoundSummaryConfig  round_summary{};

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

    int   num_exploiter_seeds = 3;

    // Hands in the post-training eval match
    int   eval_hands         = 5000;

    float bb_per_unit_reward = 10.0f;
    uint64_t seed            = std::random_device()();
};
} // namespace poker_ppo
