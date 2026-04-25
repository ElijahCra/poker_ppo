#pragma once

#include "types.h"
#include <torch/torch.h>
#include <vector>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// ActorCritic  — shared-trunk actor-critic with legal-action masking and a
//                bet-history attention encoder.
// ─────────────────────────────────────────────────────────────────────────────
//
// Observation layout (built by PokerEnvironment::compute_observation):
//   [0          : S]                static features (cards, stacks, pot, ...)
//   [S          : S + T]            history validity mask (1 = real, 0 = pad)
//   [S + T      : S + T + T*F]      history tokens, row-major (T tokens × F)
//
// Forward pass:
//   static_features  ─────────────────────────────────────┐
//                                                         │
//   history_tokens ── Linear(F→D) ── prepend CLS ──┐      │
//                                                  │      │
//                          + learnable pos-emb     │      │
//                                                  ▼      │
//                          [self-attention block × N]     │
//                                                  │      │
//                                pool = output[:, 0]      │
//                                                  ▼      ▼
//                                          concat(static, pool)
//                                                  │
//                                          [shared MLP trunk]
//                                                  │
//                                          ┌───────┴───────┐
//                                       actor          critic
//
// The actor head applies a legal-action mask before softmax:
//   masked_logits = logits + (1 - mask) * (-1e8)
//
// A single network is used for both players (parameter sharing across seats).

class ActorCriticImpl : public torch::nn::Module {
public:
    ActorCriticImpl(int obs_dim, int action_count,
                    int hidden_dim, int num_layers,
                    BetHistoryConfig hist);

    /// Forward pass.  Returns {logits, value}.
    /// logits are *unmasked* — call get_action() or evaluate() for masking.
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor obs);

    /// Sample an action with masked categorical, return {action, log_prob, value}.
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
    /// obs:     [B, obs_dim]
    /// mask:    [B, action_count]
    /// action:  [B]  int64
    struct EvalResult {
        torch::Tensor log_prob;  // [B]
        torch::Tensor value;     // [B]
        torch::Tensor entropy;   // [B]
    };
    EvalResult evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                        const torch::Tensor &action);

private:
    torch::Tensor apply_mask(torch::Tensor logits, torch::Tensor mask);

    /// Encode the bet-history slice of `obs` to a [B, attn_dim] vector.
    /// Reads the trailing T*(1+F) entries of `obs`.
    torch::Tensor encode_history(const torch::Tensor& history_block);

    // ── attention encoder ───────────────────────────────────────────────
    BetHistoryConfig hist_;
    int  static_dim_  = 0;   // obs_dim minus hist.history_block_dim()

    torch::nn::Linear token_embed_{nullptr};   // F → D

    // Learnable CLS token [1, 1, D] and positional embedding [1, T+1, D].
    torch::Tensor cls_token_;
    torch::Tensor pos_embed_;

    // One stack per block: pre-norm ─ MHA ─ residual ─ pre-norm ─ FFN ─ residual.
    std::vector<torch::nn::LayerNorm> attn_ln_;
    std::vector<torch::nn::Linear>    qkv_proj_;   // D → 3D
    std::vector<torch::nn::Linear>    out_proj_;   // D → D
    std::vector<torch::nn::LayerNorm> ffn_ln_;
    std::vector<torch::nn::Linear>    ffn1_;       // D → ffn_mult * D
    std::vector<torch::nn::Linear>    ffn2_;       // ffn_mult * D → D

    // ── shared trunk + heads ────────────────────────────────────────────
    torch::nn::Sequential trunk_{nullptr};
    torch::nn::Linear actor_head_{nullptr};
    torch::nn::Linear critic_head_{nullptr};
};

TORCH_MODULE(ActorCritic);

} // namespace poker_ppo
