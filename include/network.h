#pragma once

#include "config.h"
#include "observation_builder.h"
#include <torch/torch.h>
#include <vector>

namespace poker_ppo {

// Additive logit for illegal actions. -1e8 instead of -inf so log-probs
// stay finite if a caller gathers an illegal slot.
inline constexpr float kIllegalActionLogit = -1e8f;

// More aggressive than kIllegalActionLogit because attention's softmax
// runs over keys that are already attended; -1e8 leaks ~1e-9 weight on
// padded positions (caught by attention tests).
inline constexpr float kAttentionMaskLogit = -1e9f;

inline constexpr float kAdvantageEps = 1e-8f;

// One half of an actor-critic pair: optional bet-history attention encoder
// + MLP trunk + linear head. Used twice by ActorCritic so the value-loss
// gradient doesn't flow into the actor's representation (CleanRL convention).
class TowerImpl : public torch::nn::Module {
public:
    TowerImpl(int obs_dim, int output_dim,
              int hidden_dim, int num_layers,
              float head_init_std,
              BetHistoryConfig    hist,
              RoundSummaryConfig  round_summary);

    torch::Tensor forward(torch::Tensor obs);

private:
    torch::Tensor encode_history(const torch::Tensor& history_block);

    BetHistoryConfig    hist_;
    RoundSummaryConfig  round_summary_;
    ObservationLayout   layout_;

    // Only registered when hist.enabled.
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

// Two independent Towers (no shared params). Same instance plays both
// seats — sharing across players, not across actor/critic.
//
// Heads: actor std=0.01, critic std=1.0; trunk orthogonal(√2)+Tanh.
// Actor masks illegal actions before softmax via kIllegalActionLogit.
class ActorCriticImpl : public torch::nn::Module {
public:
    ActorCriticImpl(int obs_dim, int action_count,
                    int hidden_dim, int num_layers,
                    BetHistoryConfig    hist,
                    RoundSummaryConfig  round_summary = {});

    // {logits, value}. Logits unmasked — use get_action()/evaluate() for masking.
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor obs);

    // Critic-only. Skips the actor tower (e.g. for GAE bootstrap).
    torch::Tensor get_value(torch::Tensor obs);

    struct ActionResult {
        torch::Tensor action;    // [B] int64
        torch::Tensor log_prob;  // [B]
        torch::Tensor value;     // [B]
        torch::Tensor entropy;   // [B]
    };
    ActionResult get_action(torch::Tensor obs, torch::Tensor legal_mask);

    struct EvalResult {
        torch::Tensor log_prob;
        torch::Tensor value;
        torch::Tensor entropy;
    };
    EvalResult evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                        const torch::Tensor &action);

private:
    torch::Tensor apply_mask(torch::Tensor logits, torch::Tensor mask);

    Tower actor_{nullptr};
    Tower critic_{nullptr};
};

TORCH_MODULE(ActorCritic);

// Typed deep copy. libtorch's Module::clone() returns a base Module and
// needs param re-registration, which the pool and the BR evaluator both
// don't want. Allocates a fresh ActorCritic, copies params + buffers
// under NoGradGuard, moves to device, sets eval() (clones are always frozen).
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
