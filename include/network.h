#pragma once

#include "types.h"
#include <torch/torch.h>

namespace poker_ppo {

// ─────────────────────────────────────────────────────────────────────────────
// Actor  — independent policy network with legal-action masking
// ─────────────────────────────────────────────────────────────────────────────
//
// Architecture:
//   observation → [MLP trunk] → features → linear head → logits [action_count]
//
// The actor applies a legal-action mask before softmax:
//   masked_logits = logits + (1 - mask) * (-1e8)
//
// A single Actor is used for both players (parameter sharing across seats),
// consistent with the self-play PG approach shown effective in the paper.

class ActorImpl : public torch::nn::Module {
public:
    ActorImpl(int obs_dim, int action_count, int hidden_dim, int num_layers);

    /// Raw logits (unmasked).  Shape: [B, action_count].
    torch::Tensor forward(torch::Tensor obs);

    /// Sample an action with masked categorical.
    struct ActionResult {
        torch::Tensor action;    // [B]   int64
        torch::Tensor log_prob;  // [B]
        torch::Tensor entropy;   // [B]
    };
    ActionResult get_action(torch::Tensor obs, torch::Tensor legal_mask);

    /// Evaluate previously-taken actions (for PPO loss computation).
    struct EvalResult {
        torch::Tensor log_prob;  // [B]
        torch::Tensor entropy;   // [B]
    };
    EvalResult evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                        torch::Tensor action);

private:
    torch::Tensor apply_mask(torch::Tensor logits, torch::Tensor mask);

    torch::nn::Sequential trunk_{nullptr};
    torch::nn::Linear      head_{nullptr};
};

TORCH_MODULE(Actor);

// ─────────────────────────────────────────────────────────────────────────────
// Critic  — independent value network
// ─────────────────────────────────────────────────────────────────────────────
//
// Architecture:
//   observation → [MLP trunk] → features → linear head → value [1]
//
// Completely separate parameters from the Actor.  This eliminates gradient
// interference between the policy and value objectives — the value loss
// cannot corrupt the policy's learned features, and vice versa.

class CriticImpl : public torch::nn::Module {
public:
    CriticImpl(int obs_dim, int hidden_dim, int num_layers);

    /// Scalar value estimate.  Shape: [B].
    torch::Tensor forward(torch::Tensor obs);

private:
    torch::nn::Sequential trunk_{nullptr};
    torch::nn::Linear      head_{nullptr};
};

TORCH_MODULE(Critic);

} // namespace poker_ppo
