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

// Temporal-conv history encoder: 2× Conv1d(k=3) + masked mean-pool.
// Full receptive field at T≤16, inherently positional, ~1/6 the FLOPs of
// the transformer. Padding bleed: k=3 mixes padded slots into valid
// positions, which is deterministic only because the obs builder zeroes
// padded token slots — keep that invariant.
class ConvHistoryEncoderImpl : public torch::nn::Module {
public:
    explicit ConvHistoryEncoderImpl(BetHistoryConfig hist);
    torch::Tensor forward(const torch::Tensor& history_block);
    [[nodiscard]] int output_dim() const noexcept { return hist_.attn_dim; }

private:
    BetHistoryConfig  hist_;
    torch::nn::Conv1d conv1_{nullptr};
    torch::nn::Conv1d conv2_{nullptr};
};
TORCH_MODULE(ConvHistoryEncoder);

// Attention-pooling history encoder (PMA/Perceiver-style readout): one
// learned query cross-attends over the token embeddings, then a small
// FFN. Keeps content-weighted readout — the part of attention that
// plausibly matters at T≤16 — at ~1/8 the cost of self-attention blocks
// (the query side has 1 row instead of T+1).
class AttnPoolHistoryEncoderImpl : public torch::nn::Module {
public:
    explicit AttnPoolHistoryEncoderImpl(BetHistoryConfig hist);
    torch::Tensor forward(const torch::Tensor& history_block);
    [[nodiscard]] int output_dim() const noexcept { return hist_.attn_dim; }

private:
    BetHistoryConfig hist_;

    torch::nn::Linear token_embed_{nullptr};
    torch::Tensor     pos_embed_;
    torch::Tensor     query_;          // [1, 1, D] learned readout seed
    torch::nn::LayerNorm kv_ln_{nullptr};
    torch::nn::Linear kv_proj_{nullptr};
    torch::nn::Linear out_proj_{nullptr};
    torch::nn::LayerNorm ffn_ln_{nullptr};
    torch::nn::Linear ffn1_{nullptr};
    torch::nn::Linear ffn2_{nullptr};
};
TORCH_MODULE(AttnPoolHistoryEncoder);

// Null-hypothesis encoder: one Linear over the raw history block (mask +
// tokens, position-aligned and zero-padded). The 4×512 trunk does any
// further mixing.
class FlattenHistoryEncoderImpl : public torch::nn::Module {
public:
    explicit FlattenHistoryEncoderImpl(BetHistoryConfig hist);
    torch::Tensor forward(const torch::Tensor& history_block);
    [[nodiscard]] int output_dim() const noexcept { return hist_.attn_dim; }

private:
    BetHistoryConfig  hist_;
    torch::nn::Linear proj_{nullptr};
};
TORCH_MODULE(FlattenHistoryEncoder);

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
class ActorCriticImpl : public torch::nn::Module {
public:
    ActorCriticImpl(int obs_dim, int action_count,
                    int hidden_dim, int num_layers,
                    BetHistoryConfig    hist,
                    RoundSummaryConfig  round_summary = {});

    // {logits, critic_raw}. critic_raw is the raw critic head: [B, A] action-
    // values Q(s,·) when features::PRIVILEGED_Q_CRITIC, else [B, 1] scalar V.
    // Logits unmasked — use get_action()/evaluate() for masking.
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor obs);

    // Expected state value with masking, for the trajectory-tail bootstrap.
    // VRPO: V̄(s)=Σ_a π(a|s)Q(s,a). Else: V(s). Returns [B].
    torch::Tensor get_state_value(torch::Tensor obs, torch::Tensor legal_mask);

    // Critic-only scalar value, unmasked policy weighting (tests/play). [B].
    torch::Tensor get_value(torch::Tensor obs);

    struct ActionResult {
        torch::Tensor action;    // [B] int64
        torch::Tensor log_prob;  // [B]
        torch::Tensor value;     // [B]  VRPO: Q(s,a_taken); else V(s)
        torch::Tensor v_bar;     // [B]  VRPO: Σ_a π(a|s)Q(s,a); else == value
    };
    ActionResult get_action(torch::Tensor obs, torch::Tensor legal_mask);

    /// `log_probs_all` is the full masked log-softmax;
    /// `log_prob.gather(action)` recovers `log_prob`. Free to expose
    /// since `evaluate()` already runs `log_softmax` internally; needed
    /// by the MMD regulariser, which computes KL across all actions.
    struct EvalResult {
        torch::Tensor log_prob;       // [B]    log π(a|s) for stored a
        torch::Tensor log_probs_all;  // [B, A] full masked log-softmax
        torch::Tensor value;          // [B]
        torch::Tensor entropy;        // [B]
    };
    EvalResult evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                        const torch::Tensor &action);

    /// Masked log-softmax over the full action set, NoGrad-friendly.
    /// Used by the MMD regulariser to evaluate the frozen magnet on the
    /// same obs the live policy is updating against. Returns [B, A].
    torch::Tensor masked_log_probs(torch::Tensor obs,
                                   torch::Tensor legal_mask);

private:
    torch::Tensor apply_mask(torch::Tensor logits, torch::Tensor mask);

    // Encoder runs only when hist_.enabled. Returns an undefined tensor
    // otherwise — callers check .defined() to skip the cat.
    torch::Tensor encode_history(const torch::Tensor& obs);

    // Concatenate the tower input from obs slices + (optional) encoded
    // history. `encoded` may be detached (critic side) or live (actor side).
    torch::Tensor build_trunk_input(const torch::Tensor& obs,
                                    const torch::Tensor& encoded);

    // Critic input = actor trunk input + (VRPO) the privileged opponent-card
    // tail block. Identical to build_trunk_input when the flag is off.
    torch::Tensor build_critic_input(const torch::Tensor& obs,
                                     const torch::Tensor& encoded);

    BetHistoryConfig    hist_;
    RoundSummaryConfig  round_summary_;
    ObservationLayout   layout_;

    // Exactly one encoder is non-null when hist_.enabled, per enc_kind_
    // (config default, overridable via POKER_PPO_HISTORY_ENCODER).
    HistoryEncoderKind     enc_kind_ = HistoryEncoderKind::Attention;
    HistoryEncoder         encoder_{nullptr};
    ConvHistoryEncoder     conv_encoder_{nullptr};
    AttnPoolHistoryEncoder pool_encoder_{nullptr};
    FlattenHistoryEncoder  flat_encoder_{nullptr};
    Tower actor_{nullptr};
    Tower critic_{nullptr};
};

TORCH_MODULE(ActorCritic);

// Chain-rule decomposition of policy entropy over the poker action space:
//   H(π) = H(fold, call, Σraises) + P(raise)·H(sizes | raise)
// At size_weight=1 this equals the full entropy exactly (unit-tested).
// The second term is what makes a vanilla entropy bonus subsidise
// aggression: 12 of 14 actions are raise sizes, so spreading mass over
// them buys ~ln(12) extra entropy — the audited result is weak hands
// carrying ~67% raise mass at ~5% per size (noise, not beliefs).
// size_weight < 1 keeps fold/call/raise exploration while removing the
// raise-multiplicity subsidy. log_probs_all: [B, A] masked log-softmax.
torch::Tensor decomposed_entropy(const torch::Tensor& log_probs_all,
                                 double size_weight);

// In-place parameter+buffer copy between structurally identical networks.
// Used for the magnet refresh: keeping the destination's storage stable
// (no realloc) means captured CUDA graphs that reference it stay valid.
void copy_actor_critic_params(const ActorCritic& src, ActorCritic& dst);

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
