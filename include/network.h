#pragma once

#include "types.h"
#include <torch/torch.h>
#include <vector>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// Tower — one half of an actor-critic pair: optional bet-history attention
//         encoder + MLP trunk + linear output head. Used twice by ActorCritic
//         (once for the policy logits, once for the state value) so that the
//         value-loss gradient does not flow into the actor's representation.
//         Matches the OpenSpiel/CleanRL single-agent convention.
// ─────────────────────────────────────────────────────────────────────────────
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
    int  static_dim_  = 0;

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

// ─────────────────────────────────────────────────────────────────────────────
// ActorCritic — two independent Towers (actor + critic), no shared parameters.
// ─────────────────────────────────────────────────────────────────────────────
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

} // namespace poker_ppo
