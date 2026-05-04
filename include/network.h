#pragma once

#include "config.h"
#include "observation_layout.h"
#include <torch/torch.h>
#include <vector>

namespace poker_ppo {

// ─── Numeric constants used by the network and downstream consumers ─────
// `kIllegalActionLogit` is the additive value applied to illegal actions
// before softmax, sending their post-softmax probability to ~0. We use
// -1e8 (and not -inf) so log-prob computations stay finite if any caller
// happens to gather an illegal slot — the value is small enough that
// fp32 softmax produces 0 to many ULPs but the upstream arithmetic never
// produces a NaN.
inline constexpr float kIllegalActionLogit = -1e8f;

// Used inside the attention encoder's softmax over keys; we want a more
// negative value here because the keys are already attended *before* the
// softmax, and -1e8 turned out to leak ~1e-9 weight on padded positions
// which the unit tests caught. -1e9 is enough to drive that residual to
// 0 in fp32.
inline constexpr float kAttentionMaskLogit = -1e9f;

// Numerical-safety epsilon added to the denominator of advantage
// normalisation in the PPO update.
inline constexpr float kAdvantageEps = 1e-8f;

// Tower — one half of an actor-critic pair: optional bet-history attention
//         encoder + MLP trunk + linear output head. Used twice by ActorCritic
//         (once for the policy logits, once for the state value) so that the
//         value-loss gradient does not flow into the actor's representation.
//         Matches the OpenSpiel/CleanRL single-agent convention.
class TowerImpl : public torch::nn::Module {
public:
    TowerImpl(int obs_dim, int output_dim,
              int hidden_dim, int num_layers,
              float head_init_std,
              BetHistoryConfig    hist,
              RoundSummaryConfig  round_summary);

    /// Returns [B, output_dim].
    torch::Tensor forward(torch::Tensor obs);

private:
    torch::Tensor encode_history(const torch::Tensor& history_block);

    BetHistoryConfig    hist_;
    RoundSummaryConfig  round_summary_;
    ObservationLayout   layout_;

    // Attention encoder (only registered if hist.enabled).
    torch::nn::Linear token_embed_{nullptr};
    torch::Tensor cls_token_;
    torch::Tensor pos_embed_;
    std::vector<torch::nn::LayerNorm> attn_ln_;
    std::vector<torch::nn::Linear>    qkv_proj_;
    std::vector<torch::nn::Linear>    out_proj_;
    std::vector<torch::nn::LayerNorm> ffn_ln_;
    std::vector<torch::nn::Linear>    ffn1_;
    std::vector<torch::nn::Linear>    ffn2_;

    torch::nn::Sequential trunk_{nullptr};
    torch::nn::Linear     head_{nullptr};
};
TORCH_MODULE(Tower);

// ActorCritic — two independent Towers (actor + critic), no shared parameters.
//
// Observation layout (built by PokerEnvironment::compute_observation):
//   [0          : S]                static features (cards, stacks, pot, ...)
//   [S          : S + RS]           optional round-summary block
//   [S + RS     : S + RS + HB]      optional bet-history block (mask + tokens)
//
// Each Tower:
//   static_features ─────────────────────────────────────┐
//   round_summary?  ─────────────────────────────────────┤
//   history_tokens? ── [self-attention block × N] ── CLS ┤
//                                                        ▼
//                                                    [trunk MLP]
//                                                        │
//                                                       head
//
// The actor head is initialised with std=0.01; the critic head with std=1.0.
// Each head's MLP trunk uses orthogonal(√2) init, biases zero, Tanh activations.
//
// The actor head applies a legal-action mask before softmax:
//   masked_logits = logits + (1 - mask) * (-1e8)
//
// A single ActorCritic instance is used for both seats (parameter sharing
// across players, not across actor/critic).
class ActorCriticImpl : public torch::nn::Module {
public:
    ActorCriticImpl(int obs_dim, int action_count,
                    int hidden_dim, int num_layers,
                    BetHistoryConfig    hist,
                    RoundSummaryConfig  round_summary = {});

    /// Forward pass. Returns {logits, value}.
    /// logits are *unmasked* — call get_action() or evaluate() for masking.
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor obs);

    /// Critic-only forward. Use this when the actor's logits aren't needed
    /// (e.g. the GAE bootstrap at the end of a rollout) so the actor tower
    /// isn't run for nothing.
    torch::Tensor get_value(torch::Tensor obs);

    /// Sample an action with masked categorical, return {action, log_prob, value, entropy}.
    /// obs:  [B, obs_dim]
    /// mask: [B, action_count]
    struct ActionResult {
        torch::Tensor action;    // [B]   int64
        torch::Tensor log_prob;  // [B]
        torch::Tensor value;     // [B]
        torch::Tensor entropy;   // [B]
    };
    ActionResult get_action(torch::Tensor obs, torch::Tensor legal_mask);

    /// Evaluate stored actions (for PPO loss).
    struct EvalResult {
        torch::Tensor log_prob;  // [B]
        torch::Tensor value;     // [B]
        torch::Tensor entropy;   // [B]
    };
    EvalResult evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                        const torch::Tensor &action);

private:
    torch::Tensor apply_mask(torch::Tensor logits, torch::Tensor mask);

    Tower actor_{nullptr};
    Tower critic_{nullptr};
};

TORCH_MODULE(ActorCritic);

// clone_actor_critic — deep-copy an ActorCritic by shape + parameters.
//
// libtorch's `clone()` on a Module does NOT do what we want here — it
// returns a base `Module`, not a typed `ActorCritic`, and it requires
// parameter buffers be re-registered. The reservoir-sampling opponent
// pool and the best-response evaluator both want a typed deep copy, so
// they both implemented the same routine. Centralised here.
//
// Allocates a fresh ActorCritic of matching shape, copies parameters and
// buffers (under NoGradGuard), moves to `device`, and switches to eval()
// since a clone is always a frozen target.
[[nodiscard]] ActorCritic clone_actor_critic(
    const ActorCritic&  src,
    int                 obs_dim,
    int                 action_count,
    int                 hidden_dim,
    int                 num_layers,
    BetHistoryConfig    hist,
    RoundSummaryConfig  round_summary,
    torch::Device       device);

} // namespace poker_ppo
