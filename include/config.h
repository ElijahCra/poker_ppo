#pragma once
// Compile-time training/eval config. Edit values, rebuild.
// Game-rule config lives in Game/GameConfig.hpp.

#include "types.h"
#include "features.h"
#include "GameConfig.hpp"

#include <cstdint>
#include <random>

namespace poker_ppo {

// How the bet-history block is summarised into a [B, attn_dim] embedding.
// All kinds share the same input layout and output width, so the trunk —
// and everything downstream — is identical; only the encoder differs.
// Selectable at runtime via POKER_PPO_HISTORY_ENCODER=attn|conv|pool|flatten
// (overrides the config default; checkpoints only load under the same kind).
//   Attention  2-block self-attention transformer, CLS readout (original).
//   Conv       2× Conv1d(k=3) + masked mean-pool. ~1/6 the encoder FLOPs;
//              relies on padded token slots being zero (obs builder zeroes).
//   AttnPool   single learned query cross-attending over tokens (PMA-style):
//              keeps content-weighted readout, drops token↔token mixing.
//   Flatten    one Linear over the raw block — the null hypothesis.
enum class HistoryEncoderKind : uint8_t { Attention, Conv, AttnPool, Flatten };

// Bet-history attention encoder. Per-action features (F=8):
//   0  amount / initial_stack
//   1  amount / (2 * initial_stack)
//   2  is_my_action
//   3  is_aggressive
//   4..7  round one-hot
struct BetHistoryConfig {
    static constexpr int feat_per_action = 8;

    bool enabled         = true;
    HistoryEncoderKind kind = HistoryEncoderKind::Attention;
    int  max_history_len = 32;    // T
    int  attn_dim        = 64;    // D (= encoder output dim for all kinds),
                                  // must be divisible by attn_heads
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
// Cadences are in ENVIRONMENT STEPS (global_step), not updates, so they
// survive batch-shape changes (num_envs/num_steps) unchanged.
struct OpponentPoolConfig {
    bool     enabled                = false;
    int      max_size               = 20;
    int64_t  snapshot_every_steps   = 2'457'600;   // ≈200 updates @ 12,288-step batches
    int64_t  warmup_steps           = 2'457'600;
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

    // Weight on the raise-size term of the chain-rule entropy
    // decomposition H = H(fold,call,raise) + w·P(raise)·H(sizes|raise).
    // 1.0 = plain entropy. 0.25 adopted 2026-06-12: with 12 of 14 actions
    // being raise sizes, plain entropy pays ~ln(12) for spreading mass
    // across raises — the audited cause of weak-hand over-aggression AND
    // the dominant exploitability leak: paired 3-seed BR bound fell
    // 0.958 → 0.228 bb/hand (mean 0.743 → −0.486; exploiters lose).
    // POKER_PPO_ENTROPY_SIZE_WEIGHT overrides (1.0 restores plain).
    float entropy_size_weight = 0.25f;

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

    // ── MMD regularisation (Sokota et al., ICLR 2023) ────────────────────
    // When `kl_coef > 0`, adds `kl_coef * KL(π_θ || ρ)` to the PPO loss,
    // where ρ is a slowly-updated "magnet" snapshot of π_θ. This converts
    // vanilla self-play PPO into Magnetic Mirror Descent — a regularised
    // PG method with last-iterate Nash convergence guarantees in the
    // tabular case (and empirically in the deep variant). Cheap to add:
    // one full-batch magnet forward per update (precomputed log-probs) +
    // a clone of the policy every `magnet_refresh_steps` env steps.
    //
    // 0.05 is the value Sokota 2023 found best-on-average; the sweep
    // range was [2^-3..2^3] × that. Set kl_coef = 0 to disable (recovers
    // vanilla self-play PPO bit-for-bit).
    float kl_coef             = 0.05f;
    // R-NaD (Perolat et al. 2022): when > 0, the magnet KL moves from the
    // loss into the REWARDS — each π-controlled action perturbs rewards
    // zero-sum (actor −η(logπ−logρ), opponent +η), the critic learns the
    // regularised game, and the loss-side kl_coef term switches off.
    // 0.1 adopted via paired 3-seed BR A/B at 36.9M steps (conv encoder):
    // bound 1.545 (MMD loss) → 0.958 (η=0.1), while η=0.3 over-regularises
    // to 2.047 — a clean inverted-U, so the dose is load-bearing.
    // POKER_PPO_RNAD_ETA overrides (0 disables → MMD loss form returns).
    // Requires kl_coef > 0 (that allocates the magnet).
    float rnad_eta            = 0.1f;
    // Env steps between magnet refreshes. Sokota 2023's grid landed at
    // K ≈ 100 updates at the original 12,288-step batch ⇒ ≈1.23M steps.
    int64_t magnet_refresh_steps = 1'228'800;

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

    // All-in equity rewards: when both players are all-in before the river,
    // reward the *expected* showdown equity over remaining board run-outs
    // instead of the realised (random-card) outcome. Removes run-out
    // variance from the gradient — the standard poker-RL variance reduction.
    // Exact for flop/turn; Monte-Carlo (allin_equity_mc_samples run-outs)
    // preflop. POKER_PPO_NO_ALLIN_EQUITY=1 disables at runtime for A/B.
    bool allin_equity             = true;
    int  allin_equity_mc_samples  = 1000;

    // Base seed; each env gets seed ^ instance hash.
    uint64_t seed = std::random_device()();

    [[nodiscard]] constexpr int num_raise_slots() const noexcept { return game.num_raise_slots(); }
    [[nodiscard]] constexpr int action_count()    const noexcept { return game.action_count(); }
};

// Approximate best-response evaluator. See best_response.h.
struct BestResponseConfig {
    bool  enabled            = false;
    // Env steps between evals (batch-shape-invariant). updates_per_eval
    // stays in (exploiter) updates — the exploiter has its own fixed
    // num_envs/num_steps, independent of the learner's batch shape.
    int64_t eval_every_steps = 12'288'000;  // ≈1000 updates @ 12,288-step batches
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
    .clip_coef        = 0.1f,   // matches Rudolph et al. 2026 (ICLR) PPO default
    // Constant entropy in the 0.05–0.2 band that Rudolph et al. 2026 found
    // minimises exploitability for PG methods in imperfect-info games (their
    // default 0.05; #1-importance hyperparameter). 0.4 was above the band;
    // a 0.1-diagnostic on this game sharpened the policy without collapse.
    .ent_coef         = 0.1f,
    .vf_coef          = 0.5f,
    .max_grad_norm    = 0.5f,
    .clip_vloss       = true,
    .norm_advantages  = true,

    .learning_rate    = 3.0e-4f,
    .anneal_lr        = true,
    .min_lr_frac      = 0.2f,

    // Off: Rudolph et al. 2026 use a CONSTANT entropy coefficient ("instead
    // of a custom schedule") — annealing toward 0.01 ends in the standard-RL
    // regime they show is bad for exploitability.
    .anneal_ent_coef  = false,
    .ent_coef_min     = 0.1f,

    // 768 envs × 64 steps. batch_size = 768·64 = 49152 and minibatch = 3072
    // are IDENTICAL to the old 384×128 — so gradient noise scale, optimizer
    // steps/sample, the captured update-graph shape, and every dynamics
    // result (R-NaD, size-weight, BR bounds) are unchanged. The trade buys
    // rollout throughput only: doubling num_envs makes each step's inference
    // batch bigger (launch-bound forwards amortise better — measured ~25%
    // lower rollout ms end-to-end), and halving num_steps keeps the batch
    // fixed. The sole behavioural change is the GAE truncation horizon
    // (128→64 steps), negligible at gae_lambda=0.90 (~10-step effective
    // horizon). Use tools/bench_throughput.sh to re-find the knee on a
    // different GPU; the update half is compute-bound and unaffected by this.
    .num_envs         = 768,
    .num_steps        = 64,
    .update_epochs    = 4,
    .num_minibatches  = 16,

    .total_timesteps  = 600'000'000,

    .hidden_dim       = 512,
    .num_layers       = 4,
    .hist             = BetHistoryConfig{
        .enabled         = true,    // build gate: features::ATTENTION_ENCODER
        // Conv adopted 2026-06-11 via paired 3-seed BR-exploitability A/B
        // at 36.9M steps: conv max/mean 1.55/1.17 bb/hand vs attention's
        // 2.10/1.83 — at least equal (directionally better), at 3.2×
        // end-to-end training speed. Single-seed readings had pointed the
        // other way (attn 0.884) — that was a weak-attacker fluke; trust
        // only multi-seed bounds (POKER_PPO_BR_SEEDS).
        .kind            = HistoryEncoderKind::Conv,
        .max_history_len = 16,       // HUNL caps actions at ~16/hand
        // For conv this is the channel count. (96-vs-128 width assessment,
        // 2026-06: capacity is not the binding constraint in this setup —
        // the EV ceiling traced to imperfect info, fixed by the privileged
        // critic; tokens are 8 low-entropy features over T≤16.)
        .attn_dim        = 96,
        .attn_heads      = 4,        // attn/pool kinds only
        .ffn_mult        = 3,        // attn/pool kinds only
        .num_blocks      = 2,        // attn kind only
    },
    .round_summary    = RoundSummaryConfig{
        .enabled = true,            // build gate: features::ROUND_SUMMARY
    },

    .opp_pool         = OpponentPoolConfig{
        .enabled                 = true,
        .max_size                = 20,
        // ≈ the old 200-update snapshot / 400-update warmup cadence at the
        // original 12,288-step batch.
        .snapshot_every_steps    = 2'457'600,
        .warmup_steps            = 4'915'200,
        .p_use_pool              = 0.05f,
        .max_unique_per_rollout  = 4,
        // Fixed: pool-RNG turned out to be the dominant run-to-run noise
        // in BR-exploitability readouts (same-config reruns differed by
        // ±0.65 bb/hand with random_device seeding). Statistically
        // identical for training; pairs A/B arms.
        .seed                    = 0xA5C0FFEEull,
    },
};

static constexpr BestResponseConfig kBRConfig{
    .enabled            = true,
    // ≈ the old 3000-update cadence at the original 12,288-step batch:
    // less frequent → cheaper overall, deeper per eval.
    .eval_every_steps   = 36'864'000,
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
