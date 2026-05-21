#pragma once

#include "config.h"
#include "observation_builder.h"
#include <torch/torch.h>
#include <vector>

namespace poker_ppo {

// Additive logit for illegal actions
inline constexpr float kIllegalActionLogit = -1e8f;

// More aggressive than kIllegalActionLogit because attention's softmax
// runs over keys that are already attended
inline constexpr float kAttentionMaskLogit = -1e9f;

inline constexpr float kAdvantageEps = 1e-8f;

// Bet-history transformer encoder shared across actor and critic
// Only instantiated when hist.enabled.
class HistoryEncoderImpl : public torch::nn::Module {
public:
    explicit HistoryEncoderImpl(BetHistoryConfig hist);

    // history_block: [B, T*(1+F)] — first T floats are the validity mask,
    // followed by T*F token features. Returns the CLS embedding [B, D].
    torch::Tensor forward(const torch::Tensor& history_block);

    [[nodiscard]] int output_dim() const noexcept { return hist_.attn_dim; }

private:
    BetHistoryConfig hist_;

    torch::nn::Linear token_embed_{nullptr};
    torch::Tensor cls_token_;
    torch::Tensor pos_embed_;
    std::vector<torch::nn::LayerNorm> attn_ln_;
    std::vector<torch::nn::Linear>    qkv_proj_;
    std::vector<torch::nn::Linear>    out_proj_;
    std::vector<torch::nn::LayerNorm> ffn_ln_;
    std::vector<torch::nn::Linear>    ffn1_;
    std::vector<torch::nn::Linear>    ffn2_;
};
TORCH_MODULE(HistoryEncoder);

// MLP trunk + linear head. Used twice by ActorCritic so the value-loss
// gradient doesn't flow into the actor's representation
// Pre-flattened input: ActorCritic does the obs slicing and (optional)
// history encoding, then hands the tower a single trunk-input tensor.
class TowerImpl : public torch::nn::Module {
public:
    TowerImpl(int in_dim, int output_dim,
              int hidden_dim, int num_layers,
              float head_init_std);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Sequential trunk_{nullptr};
    torch::nn::Linear     head_{nullptr};
};
TORCH_MODULE(Tower);

// Two independent Towers (no shared MLP params). Optional shared
// HistoryEncoder is run once per forward; its output is fed to the
// actor with grad and to the critic detached, so value loss can't
// reshape the encoder's representation.
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

    // `log_probs_all` is the full masked log-softmax;
    // `log_prob.gather(action)` recovers `log_prob`.
    // needed by the MMD regulariser, which computes KL across all actions.
    struct EvalResult {
        torch::Tensor log_prob;       // [B]    log π(a|s) for stored a
        torch::Tensor log_probs_all;  // [B, A] full masked log-softmax
        torch::Tensor value;          // [B]
        torch::Tensor entropy;        // [B]
    };

    EvalResult evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                        const torch::Tensor &action);

    // Masked log-softmax over the full action set
    torch::Tensor masked_log_probs(torch::Tensor obs,
                                   torch::Tensor legal_mask);

private:
    torch::Tensor apply_mask(torch::Tensor logits, torch::Tensor mask);

    // Encoder runs only when hist_.enabled. Returns an undefined tensor
    // otherwise callers check .defined() to skip the cat.
    torch::Tensor encode_history(const torch::Tensor& obs);

    // Concatenate the tower input from obs slices + (optional) encoded
    // history. `encoded` may be detached (critic side) or live (actor side).
    torch::Tensor build_trunk_input(const torch::Tensor& obs,
                                    const torch::Tensor& encoded);

    BetHistoryConfig    hist_;
    RoundSummaryConfig  round_summary_;
    ObservationLayout   layout_;

    HistoryEncoder encoder_{nullptr};   // unset when hist_.enabled is false
    Tower actor_{nullptr};
    Tower critic_{nullptr};
};

TORCH_MODULE(ActorCritic);

// Typed deep copy. libtorch's Module::clone() returns a base Module and
// needs param re-registration, which errors in the  pool and the BR evaluator
// Allocates a fresh ActorCritic and copies params + buffers
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
