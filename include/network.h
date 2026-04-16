#pragma once

#include "types.h"
#include <torch/torch.h>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// ActorCritic  — shared-trunk actor-critic with legal-action masking
// ─────────────────────────────────────────────────────────────────────────────
//
// Architecture:
//   observation → [shared MLP trunk] → features
//       features → actor head  → logits [action_count]
//       features → critic head → value  [1]
//
// The actor head applies a legal-action mask *before* softmax:
//   masked_logits = logits + (1 - mask) * (-1e8)
//
// A single network is used for both players (parameter sharing), consistent
// with the approach shown effective in the paper for self-play PG methods.

class ActorCriticImpl : public torch::nn::Module {
public:
    ActorCriticImpl(int obs_dim, int action_count, int hidden_dim, int num_layers);

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
                        torch::Tensor action);

private:
    torch::Tensor apply_mask(torch::Tensor logits, torch::Tensor mask);

    torch::nn::Sequential trunk_{nullptr};
    torch::nn::Linear actor_head_{nullptr};
    torch::nn::Linear critic_head_{nullptr};
};

TORCH_MODULE(ActorCritic);

} // namespace poker_ppo
