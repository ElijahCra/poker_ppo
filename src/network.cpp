#include "network.h"
#include "features.h"

#include <torch/torch.h>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

namespace poker_ppo {

namespace {
// Whether the critic detaches the history encoder. Default is now to let
// the value-loss gradient flow, making the encoder a shared backbone
// trained by both heads: in this history-dependent game the value and
// policy want the same betting-history features, so the sharing is
// constructive (and an A/B showed a small early explained-variance edge
// with no destabilisation). The old PPG "policy-phase" detach is kept
// behind the env var for comparison.
//   unset / 1 / true  → gradients flow (shared encoder; default)
//   0 / false         → detach (value loss can't reshape the encoder)
bool critic_detaches_encoder() {
    static const bool detaches = [] {
        const char* e = std::getenv("POKER_PPO_CRITIC_ENCODER_GRAD");
        if (!e) return false;  // default: gradients flow
        const std::string v(e);
        return v == "0" || v == "false";
    }();
    return detaches;
}

// History-encoder kind: config default, overridable via env for A/B runs
// without a rebuild. Only the ENV override is cached (read once), so every
// network in a training process (learner, magnet, pool snapshots, BR
// exploiter, league) agrees under an override, while tests can still
// construct different kinds via cfg.kind in one process. A checkpoint only
// loads under the kind it was trained with.
HistoryEncoderKind history_encoder_kind(HistoryEncoderKind def) {
    static const std::optional<HistoryEncoderKind> env_kind =
        []() -> std::optional<HistoryEncoderKind> {
        const char* e = std::getenv("POKER_PPO_HISTORY_ENCODER");
        if (!e) return std::nullopt;
        const std::string v(e);
        if (v == "attn")    return HistoryEncoderKind::Attention;
        if (v == "conv")    return HistoryEncoderKind::Conv;
        if (v == "pool")    return HistoryEncoderKind::AttnPool;
        if (v == "flatten") return HistoryEncoderKind::Flatten;
        std::cerr << "[network] unknown POKER_PPO_HISTORY_ENCODER='" << v
                  << "' (want attn|conv|pool|flatten); using config default\n";
        return std::nullopt;
    }();
    return env_kind.value_or(def);
}

void xavier_lin_init(torch::nn::Linear& lin) {
    torch::nn::init::xavier_uniform_(lin->weight);
    torch::nn::init::constant_(lin->bias, 0.0);
}
}  // namespace

// ─── HistoryEncoder ──────────────────────────────────────────────────────

HistoryEncoderImpl::HistoryEncoderImpl(BetHistoryConfig hist)
    : hist_(hist)
{
    const int D  = hist_.attn_dim;
    const int H  = hist_.attn_heads;
    const int T  = hist_.max_history_len;
    const int F  = BetHistoryConfig::feat_per_action;
    const int FF = hist_.ffn_mult * D;

    if (D <= 0 || H <= 0 || D % H != 0) {
        throw std::invalid_argument(
            "BetHistoryConfig: attn_dim must be a positive multiple of attn_heads");
    }

    token_embed_ = register_module("token_embed", torch::nn::Linear(F, D));

    cls_token_ = register_parameter("cls_token", torch::zeros({1, 1, D}));
    pos_embed_ = register_parameter("pos_embed", torch::zeros({1, T + 1, D}));
    torch::nn::init::normal_(cls_token_, /*mean=*/0.0, /*std=*/0.02);
    torch::nn::init::normal_(pos_embed_, /*mean=*/0.0, /*std=*/0.02);

    attn_ln_.reserve(hist_.num_blocks);
    qkv_proj_.reserve(hist_.num_blocks);
    out_proj_.reserve(hist_.num_blocks);
    ffn_ln_.reserve(hist_.num_blocks);
    ffn1_.reserve(hist_.num_blocks);
    ffn2_.reserve(hist_.num_blocks);
    for (int b = 0; b < hist_.num_blocks; ++b) {
        const std::string s = std::to_string(b);
        attn_ln_.push_back(register_module(
            "attn_ln_" + s, torch::nn::LayerNorm(torch::nn::LayerNormOptions({D}))));
        qkv_proj_.push_back(register_module(
            "qkv_proj_" + s, torch::nn::Linear(D, 3 * D)));
        out_proj_.push_back(register_module(
            "out_proj_" + s, torch::nn::Linear(D, D)));
        ffn_ln_.push_back(register_module(
            "ffn_ln_" + s, torch::nn::LayerNorm(torch::nn::LayerNormOptions({D}))));
        ffn1_.push_back(register_module(
            "ffn1_" + s, torch::nn::Linear(D, FF)));
        ffn2_.push_back(register_module(
            "ffn2_" + s, torch::nn::Linear(FF, D)));
    }

    auto xavier_lin = [](torch::nn::Linear& lin) {
        torch::nn::init::xavier_uniform_(lin->weight);
        torch::nn::init::constant_(lin->bias, 0.0);
    };
    xavier_lin(token_embed_);
    for (int b = 0; b < hist_.num_blocks; ++b) {
        xavier_lin(qkv_proj_[b]);
        xavier_lin(out_proj_[b]);
        xavier_lin(ffn1_[b]);
        xavier_lin(ffn2_[b]);
    }
}

torch::Tensor
HistoryEncoderImpl::forward(const torch::Tensor& history_block) {
    // history_block: [B, T*(1+F)]
    const int T = hist_.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;
    const int D = hist_.attn_dim;
    const int H = hist_.attn_heads;
    const int d_head = D / H;
    const int64_t B = history_block.size(0);

    auto mask   = history_block.narrow(/*dim=*/1, /*start=*/0, /*length=*/T);
    auto tokens = history_block.narrow(1, T, T * F).reshape({B, T, F});

    auto x = token_embed_->forward(tokens);  // [B, T, D]

    auto cls = cls_token_.expand({B, 1, D});
    x = torch::cat({cls, x}, /*dim=*/1);                            // [B, T+1, D]
    auto cls_mask  = torch::ones({B, 1}, mask.options());
    auto attn_mask = torch::cat({cls_mask, mask}, /*dim=*/1);        // [B, T+1]

    x = x + pos_embed_;

    // Additive mask: [B, 1, 1, T+1] broadcasts to [B, H, T+1, T+1] inside SDPA.
    auto add_mask = (1.0 - attn_mask).unsqueeze(1).unsqueeze(1) * kAttentionMaskLogit;

    const int64_t L = T + 1;
    for (int b = 0; b < hist_.num_blocks; ++b) {
        auto h = attn_ln_[b]->forward(x);
        auto qkv = qkv_proj_[b]->forward(h);          // [B, L, 3D]
        auto qkv_split = qkv.chunk(3, /*dim=*/-1);
        auto reshape_heads = [&](torch::Tensor t) {
            return t.reshape({B, L, H, d_head}).transpose(1, 2);
        };
        auto q = reshape_heads(qkv_split[0]);
        auto k = reshape_heads(qkv_split[1]);
        auto v = reshape_heads(qkv_split[2]);

        // Fused kernel: Flash on CUDA, oneDNN on CPU. Replaces a manual
        // matmul → scale → mask-add → softmax → matmul chain.
        auto out = torch::scaled_dot_product_attention(
            q, k, v,
            /*attn_mask=*/add_mask,
            /*dropout_p=*/0.0,
            /*is_causal=*/false);

        out = out.transpose(1, 2).contiguous().reshape({B, L, D});
        out = out_proj_[b]->forward(out);
        x = x + out;

        auto fh = ffn_ln_[b]->forward(x);
        auto ff = ffn1_[b]->forward(fh);
        ff = torch::gelu(ff);
        ff = ffn2_[b]->forward(ff);
        x = x + ff;
    }

    return x.select(/*dim=*/1, /*index=*/0);   // CLS, [B, D]
}

// ─── ConvHistoryEncoder ──────────────────────────────────────────────────

ConvHistoryEncoderImpl::ConvHistoryEncoderImpl(BetHistoryConfig hist)
    : hist_(hist)
{
    const int C = hist_.attn_dim;
    const int F = BetHistoryConfig::feat_per_action;

    conv1_ = register_module("conv1", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(F, C, /*kernel=*/3).padding(1)));
    conv2_ = register_module("conv2", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(C, C, /*kernel=*/3).padding(1)));

    for (auto* conv : {&conv1_, &conv2_}) {
        torch::nn::init::xavier_uniform_((*conv)->weight);
        torch::nn::init::constant_((*conv)->bias, 0.0);
    }
}

torch::Tensor
ConvHistoryEncoderImpl::forward(const torch::Tensor& history_block) {
    const int T = hist_.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;
    const int64_t B = history_block.size(0);

    auto mask   = history_block.narrow(1, 0, T);                    // [B, T]
    auto tokens = history_block.narrow(1, T, T * F)
                      .reshape({B, T, F})
                      .transpose(1, 2);                             // [B, F, T]

    auto x = torch::gelu(conv1_->forward(tokens));
    x      = torch::gelu(conv2_->forward(x));                       // [B, C, T]

    // Masked mean-pool over time. Padded positions contribute zero; the
    // clamp keeps the empty-history (preflop-first-action) row finite.
    auto m   = mask.unsqueeze(1);                                   // [B, 1, T]
    auto sum = (x * m).sum(/*dim=*/-1);                             // [B, C]
    auto cnt = m.sum(/*dim=*/-1).clamp_min(1.0);                    // [B, 1]
    return sum / cnt;
}

// ─── AttnPoolHistoryEncoder ──────────────────────────────────────────────

AttnPoolHistoryEncoderImpl::AttnPoolHistoryEncoderImpl(BetHistoryConfig hist)
    : hist_(hist)
{
    const int D  = hist_.attn_dim;
    const int H  = hist_.attn_heads;
    const int T  = hist_.max_history_len;
    const int F  = BetHistoryConfig::feat_per_action;
    const int FF = hist_.ffn_mult * D;

    if (D <= 0 || H <= 0 || D % H != 0) {
        throw std::invalid_argument(
            "BetHistoryConfig: attn_dim must be a positive multiple of attn_heads");
    }

    token_embed_ = register_module("token_embed", torch::nn::Linear(F, D));
    pos_embed_   = register_parameter("pos_embed", torch::zeros({1, T, D}));
    query_       = register_parameter("query",     torch::zeros({1, 1, D}));
    torch::nn::init::normal_(pos_embed_, /*mean=*/0.0, /*std=*/0.02);
    torch::nn::init::normal_(query_,     /*mean=*/0.0, /*std=*/0.02);

    kv_ln_    = register_module("kv_ln",
                    torch::nn::LayerNorm(torch::nn::LayerNormOptions({D})));
    kv_proj_  = register_module("kv_proj",  torch::nn::Linear(D, 2 * D));
    out_proj_ = register_module("out_proj", torch::nn::Linear(D, D));
    ffn_ln_   = register_module("ffn_ln",
                    torch::nn::LayerNorm(torch::nn::LayerNormOptions({D})));
    ffn1_     = register_module("ffn1", torch::nn::Linear(D, FF));
    ffn2_     = register_module("ffn2", torch::nn::Linear(FF, D));

    xavier_lin_init(token_embed_);
    xavier_lin_init(kv_proj_);
    xavier_lin_init(out_proj_);
    xavier_lin_init(ffn1_);
    xavier_lin_init(ffn2_);
}

torch::Tensor
AttnPoolHistoryEncoderImpl::forward(const torch::Tensor& history_block) {
    const int T = hist_.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;
    const int D = hist_.attn_dim;
    const int H = hist_.attn_heads;
    const int d_head = D / H;
    const int64_t B = history_block.size(0);

    auto mask   = history_block.narrow(1, 0, T);                    // [B, T]
    auto tokens = history_block.narrow(1, T, T * F).reshape({B, T, F});

    auto x = token_embed_->forward(tokens) + pos_embed_;            // [B, T, D]

    auto kv       = kv_proj_->forward(kv_ln_->forward(x)).chunk(2, -1);
    auto reshape_heads = [&](torch::Tensor t) {
        return t.reshape({B, T, H, d_head}).transpose(1, 2);        // [B, H, T, dh]
    };
    auto k = reshape_heads(kv[0]);
    auto v = reshape_heads(kv[1]);
    auto q = query_.expand({B, 1, D})
                 .reshape({B, 1, H, d_head}).transpose(1, 2);       // [B, H, 1, dh]

    // Fully-masked rows (empty history) degrade to uniform attention over
    // f(pos_embed) — a learned "no history yet" constant, never NaN
    // (equal finite logits, not -inf).
    auto add_mask = (1.0 - mask).unsqueeze(1).unsqueeze(1) * kAttentionMaskLogit;

    auto out = torch::scaled_dot_product_attention(
        q, k, v, /*attn_mask=*/add_mask, /*dropout_p=*/0.0, /*is_causal=*/false);

    auto y = out_proj_->forward(
        out.transpose(1, 2).reshape({B, D}));                       // [B, D]
    auto f = ffn2_->forward(torch::gelu(ffn1_->forward(ffn_ln_->forward(y))));
    return y + f;
}

// ─── FlattenHistoryEncoder ───────────────────────────────────────────────

FlattenHistoryEncoderImpl::FlattenHistoryEncoderImpl(BetHistoryConfig hist)
    : hist_(hist)
{
    const int in_dim = hist_.history_block_dim();
    proj_ = register_module("proj", torch::nn::Linear(in_dim, hist_.attn_dim));
    xavier_lin_init(proj_);
}

torch::Tensor
FlattenHistoryEncoderImpl::forward(const torch::Tensor& history_block) {
    // Tanh matches the trunk's activation family; the trunk does the rest.
    return torch::tanh(proj_->forward(history_block));
}

// ─── Tower (MLP trunk + head) ────────────────────────────────────────────

TowerImpl::TowerImpl(int in_dim, int output_dim,
                     int hidden_dim, int num_layers,
                     float head_init_std)
{
    torch::nn::Sequential trunk;
    int d = in_dim;
    for (int i = 0; i < num_layers; ++i) {
        trunk->push_back(torch::nn::Linear(d, hidden_dim));
        trunk->push_back(torch::nn::Tanh());
        d = hidden_dim;
    }
    trunk_ = register_module("trunk", trunk);
    head_  = register_module("head",  torch::nn::Linear(hidden_dim, output_dim));

    for (auto& m : trunk_->modules(/*include_self=*/false)) {
        if (auto* lin = m->as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, std::sqrt(2.0));
            torch::nn::init::constant_(lin->bias, 0.0);
        }
    }
    torch::nn::init::orthogonal_(head_->weight, head_init_std);
    torch::nn::init::constant_(head_->bias, 0.0);
}

torch::Tensor TowerImpl::forward(torch::Tensor x) {
    return head_->forward(trunk_->forward(x));
}

// ─── ActorCritic (shared encoder + two towers) ───────────────────────────

ActorCriticImpl::ActorCriticImpl(int obs_dim, int action_count,
                                 int hidden_dim, int num_layers,
                                 BetHistoryConfig    hist,
                                 RoundSummaryConfig  round_summary)
    : hist_(hist),
      round_summary_(round_summary),
      layout_(ObservationLayout::build(hist, round_summary))
{
    if (layout_.total_dim != obs_dim) {
        throw std::invalid_argument(
            "ActorCritic: obs_dim does not match ObservationLayout::build(hist, round_summary)");
    }

    int trunk_in_dim =
        layout_.static_off + ObservationLayout::FEAT_STATIC + layout_.round_summary_dim;

    if constexpr (features::ATTENTION_ENCODER) {
        if (hist_.enabled) {
            enc_kind_ = history_encoder_kind(hist_.kind);
            switch (enc_kind_) {
                case HistoryEncoderKind::Attention:
                    encoder_ = register_module("history_encoder",
                                               HistoryEncoder(hist_));
                    break;
                case HistoryEncoderKind::Conv:
                    conv_encoder_ = register_module("history_encoder_conv",
                                                    ConvHistoryEncoder(hist_));
                    break;
                case HistoryEncoderKind::AttnPool:
                    pool_encoder_ = register_module("history_encoder_pool",
                                                    AttnPoolHistoryEncoder(hist_));
                    break;
                case HistoryEncoderKind::Flatten:
                    flat_encoder_ = register_module("history_encoder_flat",
                                                    FlattenHistoryEncoder(hist_));
                    break;
            }
            // All kinds emit [B, attn_dim].
            trunk_in_dim += hist_.attn_dim;
        }
    }

    actor_  = register_module("actor",
                              Tower(trunk_in_dim, action_count,
                                    hidden_dim, num_layers,
                                    /*head_init_std=*/0.01f));
    // VRPO: action-value critic Q(s,·) fed the privileged opponent-card
    // tail. Otherwise a scalar V head. head_init_std=1.0 either way.
    critic_ = register_module("critic",
                              Tower(trunk_in_dim + layout_.privileged_dim,
                                    features::PRIVILEGED_Q_CRITIC ? action_count : 1,
                                    hidden_dim, num_layers,
                                    /*head_init_std=*/1.0f));
}

torch::Tensor
ActorCriticImpl::encode_history(const torch::Tensor& obs) {
    if constexpr (features::ATTENTION_ENCODER) {
        if (hist_.enabled) {
            auto history_blk = obs.narrow(1, layout_.history_off,
                                          layout_.history_dim);
            switch (enc_kind_) {
                case HistoryEncoderKind::Attention:
                    return encoder_->forward(history_blk);
                case HistoryEncoderKind::Conv:
                    return conv_encoder_->forward(history_blk);
                case HistoryEncoderKind::AttnPool:
                    return pool_encoder_->forward(history_blk);
                case HistoryEncoderKind::Flatten:
                    return flat_encoder_->forward(history_blk);
            }
        }
    }
    return {};
}

torch::Tensor
ActorCriticImpl::build_trunk_input(const torch::Tensor& obs,
                                   const torch::Tensor& encoded)
{
    const int tower_static_dim =
        layout_.static_off + ObservationLayout::FEAT_STATIC;

    auto static_feats = obs.narrow(/*dim=*/1, /*start=*/0, tower_static_dim);

    std::vector<torch::Tensor> parts;
    parts.reserve(3);
    parts.push_back(static_feats);

    if constexpr (features::ROUND_SUMMARY) {
        if (round_summary_.enabled) {
            parts.push_back(obs.narrow(1, layout_.round_summary_off,
                                       layout_.round_summary_dim));
        }
    }

    if (encoded.defined()) {
        parts.push_back(encoded);
    }

    return (parts.size() == 1) ? parts[0] : torch::cat(parts, /*dim=*/-1);
}

torch::Tensor
ActorCriticImpl::build_critic_input(const torch::Tensor& obs,
                                    const torch::Tensor& encoded) {
    auto base = build_trunk_input(obs, encoded);
    if constexpr (features::PRIVILEGED_Q_CRITIC) {
        auto priv = obs.narrow(/*dim=*/1, layout_.privileged_off,
                               layout_.privileged_dim);
        return torch::cat({base, priv}, /*dim=*/-1);
    }
    return base;
}

std::pair<torch::Tensor, torch::Tensor>
ActorCriticImpl::forward(torch::Tensor obs) {
    auto encoded = encode_history(obs);

    // By default the value-loss gradient flows into the encoder (shared
    // backbone trained by both heads). POKER_PPO_CRITIC_ENCODER_GRAD=0
    // restores the PPG policy-phase detach for comparison.
    auto actor_in = build_trunk_input(obs, encoded);
    auto critic_enc = (encoded.defined() && critic_detaches_encoder())
        ? encoded.detach()
        : encoded;
    auto critic_in = build_critic_input(obs, critic_enc);

    auto logits     = actor_->forward(actor_in);
    auto critic_raw = critic_->forward(critic_in);  // [B, A] (Q) or [B, 1] (V)
    return {logits, critic_raw};
}

torch::Tensor ActorCriticImpl::actor_logits(torch::Tensor obs) {
    auto encoded = encode_history(obs);
    return actor_->forward(build_trunk_input(obs, encoded));
}

torch::Tensor ActorCriticImpl::critic_values(torch::Tensor obs) {
    auto encoded = encode_history(obs);
    auto critic_enc = (encoded.defined() && critic_detaches_encoder())
        ? encoded.detach() : encoded;
    return critic_->forward(build_critic_input(obs, critic_enc));
}

torch::Tensor ActorCriticImpl::get_value(torch::Tensor obs) {
    // Unmasked expected value (tests/play); the masked bootstrap form is
    // get_state_value.
    auto [logits, critic_raw] = forward(obs);
    if constexpr (features::PRIVILEGED_Q_CRITIC) {
        auto dist = torch::softmax(logits, -1);
        return (dist * critic_raw).sum(-1);
    }
    return critic_raw.squeeze(-1);
}

torch::Tensor
ActorCriticImpl::get_state_value(torch::Tensor obs, torch::Tensor legal_mask) {
    auto [logits, critic_raw] = forward(obs);
    if constexpr (features::PRIVILEGED_Q_CRITIC) {
        auto dist = torch::softmax(apply_mask(logits, legal_mask), -1);
        return (dist * critic_raw).sum(-1);  // V̄(s) = Σ_a π(a|s) Q(s,a)
    }
    return critic_raw.squeeze(-1);
}

torch::Tensor
ActorCriticImpl::apply_mask(torch::Tensor logits, torch::Tensor mask) {
    return logits + (1.0f - mask) * kIllegalActionLogit;
}

ActorCriticImpl::ActionResult
ActorCriticImpl::get_action(torch::Tensor obs, torch::Tensor legal_mask) {
    auto [logits, critic_raw] = forward(obs);
    auto masked = apply_mask(logits, legal_mask);

    auto dist    = torch::softmax(masked, /*dim=*/-1);
    auto action  = dist.multinomial(/*num_samples=*/1, /*replacement=*/true)
                       .squeeze(-1);

    auto log_dist = torch::log_softmax(masked, -1);
    auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);

    torch::Tensor value, v_bar;
    if constexpr (features::PRIVILEGED_Q_CRITIC) {
        // dist is ~0 on illegal actions, so V̄ sums over legal actions only.
        v_bar = (dist * critic_raw).sum(-1);                              // V̄(s)
        value = critic_raw.gather(-1, action.unsqueeze(-1)).squeeze(-1);  // Q(s,a)
    } else {
        value = critic_raw.squeeze(-1);
        v_bar = value;
    }

    return {action, log_prob, value, v_bar};
}

ActorCriticImpl::EvalResult
ActorCriticImpl::evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                          const torch::Tensor &action) {
    auto [logits, critic_raw] = forward(obs);
    const auto masked = apply_mask(logits, legal_mask);

    const auto dist     = torch::softmax(masked, -1);
    const auto log_dist = torch::log_softmax(masked, -1);
    const auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    const auto entropy  = -(dist * log_dist).sum(-1);

    // VRPO: regress Q(s,a_taken); else the scalar V. Both shape [B].
    torch::Tensor value;
    if constexpr (features::PRIVILEGED_Q_CRITIC) {
        value = critic_raw.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    } else {
        value = critic_raw.squeeze(-1);
    }

    return {log_prob, log_dist, value, entropy};
}

torch::Tensor
ActorCriticImpl::masked_log_probs(torch::Tensor obs, torch::Tensor legal_mask) {
    // Actor-only forward — saves the critic tower call. Encoder still
    // runs once (it's not duplicated across towers anymore).
    auto encoded  = encode_history(obs);
    auto actor_in = build_trunk_input(obs, encoded);
    auto logits   = actor_->forward(actor_in);
    return torch::log_softmax(apply_mask(logits, legal_mask), -1);
}

torch::Tensor decomposed_entropy(const torch::Tensor& log_probs_all,
                                 double size_weight) {
    auto p       = log_probs_all.exp();                       // [B, A]
    auto p_f     = p.narrow(-1, 0, 1);
    auto p_c     = p.narrow(-1, 1, 1);
    auto p_sizes = p.narrow(-1, 2, p.size(-1) - 2);
    auto p_r     = p_sizes.sum(-1, /*keepdim=*/true);

    // clamp_min keeps 0·log0 = 0 for masked-out (≈zero-prob) actions.
    auto type_p = torch::cat({p_f, p_c, p_r}, -1);
    auto h_type = -(type_p * type_p.clamp_min(1e-12f).log()).sum(-1);
    auto h_size = -(p_sizes * (p_sizes.clamp_min(1e-12f).log()
                               - p_r.clamp_min(1e-12f).log())).sum(-1);
    return h_type + size_weight * h_size;
}

void copy_actor_critic_params(const ActorCritic& src, ActorCritic& dst) {
    torch::NoGradGuard ng;
    auto sp = src->parameters();
    auto dp = dst->parameters();
    TORCH_CHECK(sp.size() == dp.size(),
                "copy_actor_critic_params: parameter count mismatch");
    for (size_t i = 0; i < sp.size(); ++i) {
        dp[i].copy_(sp[i]);  // copy_ handles cross-device
    }
    auto sb = src->buffers();
    auto db = dst->buffers();
    TORCH_CHECK(sb.size() == db.size(),
                "copy_actor_critic_params: buffer count mismatch");
    for (size_t i = 0; i < sb.size(); ++i) {
        db[i].copy_(sb[i]);
    }
}

ActorCritic clone_actor_critic(const ActorCritic&  src,
                               int                 obs_dim,
                               int                 action_count,
                               int                 hidden_dim,
                               int                 num_layers,
                               BetHistoryConfig    hist,
                               RoundSummaryConfig  round_summary,
                               torch::Device       device)
{
    ActorCritic dst(obs_dim, action_count, hidden_dim, num_layers,
                    hist, round_summary);
    dst->to(device);
    copy_actor_critic_params(src, dst);
    dst->eval();
    return dst;
}

} // namespace poker_ppo
