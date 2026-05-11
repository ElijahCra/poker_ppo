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

// Bet-history transformer encoder. Owned by ActorCritic and shared
// across actor and critic — running it once per forward halves the
// attention compute compared to giving each tower its own encoder.
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
// gradient doesn't flow into the actor's representation (CleanRL convention).
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
//
// Optional auxiliary CFV head (when cfv_aux.enabled): a third Tower
// outputting [B, kCFVHeadDim] counterfactual values per hole-card combo.
// Shares the live encoder output (gradients flow back into encoder),
// providing range-aware supervision to the trunk + encoder.
class ActorCriticImpl : public torch::nn::Module {
public:
    ActorCriticImpl(int obs_dim, int action_count,
                    int hidden_dim, int num_layers,
                    BetHistoryConfig    hist,
                    RoundSummaryConfig  round_summary = {},
                    CFVAuxConfig        cfv_aux = {});

    // {logits, value}. Logits unmasked — use get_action()/evaluate() for masking.
    // CFV head NOT computed on this path — rollout doesn't need it.
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

    /// `log_probs_all` is the full masked log-softmax;
    /// `log_prob.gather(action)` recovers `log_prob`. Free to expose
    /// since `evaluate()` already runs `log_softmax` internally; needed
    /// by the MMD regulariser, which computes KL across all actions.
    /// `cfv` is undefined (.defined() == false) when the aux head is off.
    struct EvalResult {
        torch::Tensor log_prob;       // [B]    log π(a|s) for stored a
        torch::Tensor log_probs_all;  // [B, A] full masked log-softmax
        torch::Tensor value;          // [B]
        torch::Tensor entropy;        // [B]
        torch::Tensor cfv;            // [B, kCFVHeadDim] or undefined
    };
    EvalResult evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                        const torch::Tensor &action);

    /// Masked log-softmax over the full action set, NoGrad-friendly.
    /// Used by the MMD regulariser to evaluate the frozen magnet on the
    /// same obs the live policy is updating against. Returns [B, A].
    torch::Tensor masked_log_probs(torch::Tensor obs,
                                   torch::Tensor legal_mask);

    [[nodiscard]] bool has_cfv() const noexcept { return !cfv_.is_empty(); }

private:
    torch::Tensor apply_mask(torch::Tensor logits, torch::Tensor mask);

    // Encoder runs only when hist_.enabled. Returns an undefined tensor
    // otherwise — callers check .defined() to skip the cat.
    torch::Tensor encode_history(const torch::Tensor& obs);

    // Concatenate the tower input from obs slices + (optional) encoded
    // history. `encoded` may be detached (critic side) or live (actor side).
    torch::Tensor build_trunk_input(const torch::Tensor& obs,
                                    const torch::Tensor& encoded);

    BetHistoryConfig    hist_;
    RoundSummaryConfig  round_summary_;
    CFVAuxConfig        cfv_aux_;
    ObservationLayout   layout_;

    HistoryEncoder encoder_{nullptr};   // unset when hist_.enabled is false
    Tower actor_{nullptr};
    Tower critic_{nullptr};
    Tower cfv_{nullptr};                // unset when cfv_aux_.enabled is false
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
    torch::Device       device,
    CFVAuxConfig        cfv_aux = {});

} // namespace poker_ppo
