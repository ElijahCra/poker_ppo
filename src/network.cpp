#include "network.h"
#include <torch/torch.h>
#include <stdexcept>

namespace poker_ppo {

namespace {
constexpr float kNegInfMask = -1e9f;  // additive mask for masked-out keys
} // namespace

// ─────────────────────────────────────────────────────────────────────────────
ActorCriticImpl::ActorCriticImpl(int obs_dim, int action_count,
                                 int hidden_dim, int num_layers,
                                 BetHistoryConfig hist)
    : hist_(hist)
{
    const int D = hist_.attn_dim;
    const int H = hist_.attn_heads;
    const int T = hist_.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;
    const int FF = hist_.ffn_mult * D;

    if (D <= 0 || H <= 0 || D % H != 0) {
        throw std::invalid_argument(
            "BetHistoryConfig: attn_dim must be a positive multiple of attn_heads");
    }
    static_dim_ = obs_dim - hist_.history_block_dim();
    if (static_dim_ <= 0) {
        throw std::invalid_argument(
            "ActorCritic: obs_dim is smaller than the history block size");
    }

    // ── attention encoder ───────────────────────────────────────────────
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

    // ── shared MLP trunk ────────────────────────────────────────────────
    // The attention pool (D dims) is concatenated with the static features
    // (S dims) before entering the trunk.
    torch::nn::Sequential trunk;
    int in_dim = static_dim_ + D;
    for (int i = 0; i < num_layers; ++i) {
        trunk->push_back(torch::nn::Linear(in_dim, hidden_dim));
        trunk->push_back(torch::nn::Tanh());
        in_dim = hidden_dim;
    }
    trunk_ = register_module("trunk", trunk);

    // Actor and critic heads
    actor_head_  = register_module("actor_head",
                                   torch::nn::Linear(hidden_dim, action_count));
    critic_head_ = register_module("critic_head",
                                   torch::nn::Linear(hidden_dim, 1));

    // ── init: orthogonal for trunk/heads, xavier for attention ──────────
    for (auto& m : trunk_->modules(/*include_self=*/false)) {
        if (auto* lin = m->as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, std::sqrt(2.0));
            torch::nn::init::constant_(lin->bias, 0.0);
        }
    }
    torch::nn::init::orthogonal_(actor_head_->weight, 0.01);
    torch::nn::init::constant_(actor_head_->bias, 0.0);
    torch::nn::init::orthogonal_(critic_head_->weight, 1.0);
    torch::nn::init::constant_(critic_head_->bias, 0.0);

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

// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor
ActorCriticImpl::encode_history(const torch::Tensor& history_block) {
    // history_block: [B, T*(1+F)]
    const int T = hist_.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;
    const int D = hist_.attn_dim;
    const int H = hist_.attn_heads;
    const int d_head = D / H;
    const int64_t B = history_block.size(0);

    // Slice into mask [B, T] and tokens [B, T, F].
    auto mask   = history_block.narrow(/*dim=*/1, /*start=*/0, /*length=*/T);
    auto tokens = history_block.narrow(1, T, T * F).reshape({B, T, F});

    auto x = token_embed_->forward(tokens);  // [B, T, D]

    // Prepend CLS token (always-valid) and matching mask entry.
    auto cls = cls_token_.expand({B, 1, D});
    x = torch::cat({cls, x}, /*dim=*/1);                            // [B, T+1, D]
    auto cls_mask = torch::ones({B, 1}, mask.options());
    auto attn_mask = torch::cat({cls_mask, mask}, /*dim=*/1);        // [B, T+1]

    x = x + pos_embed_;  // broadcast [1, T+1, D] over batch

    // Pre-computed additive mask for keys: 0 where valid, -inf where padded.
    // Shape [B, 1, 1, T+1] so it broadcasts over (heads, queries).
    auto add_mask = (1.0 - attn_mask).unsqueeze(1).unsqueeze(1) * kNegInfMask;

    const int64_t L = T + 1;
    for (int b = 0; b < hist_.num_blocks; ++b) {
        // ── multi-head self-attention (pre-norm + residual) ─────────────
        auto h = attn_ln_[b]->forward(x);
        auto qkv = qkv_proj_[b]->forward(h);          // [B, L, 3D]
        auto qkv_split = qkv.chunk(3, /*dim=*/-1);
        // [B, H, L, d_head]
        auto reshape_heads = [&](torch::Tensor t) {
            return t.reshape({B, L, H, d_head}).transpose(1, 2).contiguous();
        };
        auto q = reshape_heads(qkv_split[0]);
        auto k = reshape_heads(qkv_split[1]);
        auto v = reshape_heads(qkv_split[2]);

        auto scores = torch::matmul(q, k.transpose(-2, -1))
                    / std::sqrt(static_cast<double>(d_head));         // [B, H, L, L]
        scores = scores + add_mask;
        auto attn = torch::softmax(scores, /*dim=*/-1);

        auto out = torch::matmul(attn, v);                            // [B, H, L, d_h]
        out = out.transpose(1, 2).contiguous().reshape({B, L, D});    // [B, L, D]
        out = out_proj_[b]->forward(out);
        x = x + out;

        // ── feed-forward (pre-norm + residual) ──────────────────────────
        auto fh = ffn_ln_[b]->forward(x);
        auto ff = ffn1_[b]->forward(fh);
        ff = torch::gelu(ff);
        ff = ffn2_[b]->forward(ff);
        x = x + ff;
    }

    // CLS output is the pooled history representation.
    return x.select(/*dim=*/1, /*index=*/0);   // [B, D]
}

// ─────────────────────────────────────────────────────────────────────────────
std::pair<torch::Tensor, torch::Tensor>
ActorCriticImpl::forward(torch::Tensor obs) {
    // Split obs into [B, S] static and [B, T*(1+F)] history.
    auto static_feats = obs.narrow(/*dim=*/1, /*start=*/0, /*length=*/static_dim_);
    auto history_blk  = obs.narrow(/*dim=*/1, static_dim_, hist_.history_block_dim());

    auto hist_pool = encode_history(history_blk);                      // [B, D]
    auto features  = trunk_->forward(
        torch::cat({static_feats, hist_pool}, /*dim=*/-1));            // [B, hidden]

    auto logits = actor_head_->forward(features);
    auto value  = critic_head_->forward(features).squeeze(-1);
    return {logits, value};
}

// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor
ActorCriticImpl::apply_mask(torch::Tensor logits, torch::Tensor mask) {
    // mask: 1 = legal, 0 = illegal
    return logits + (1.0f - mask) * (-1e8f);
}

// ─────────────────────────────────────────────────────────────────────────────
ActorCriticImpl::ActionResult
ActorCriticImpl::get_action(torch::Tensor obs, torch::Tensor legal_mask) {
    auto [logits, value] = forward(obs);
    auto masked = apply_mask(logits, legal_mask);

    auto dist    = torch::softmax(masked, /*dim=*/-1);
    auto action  = dist.multinomial(/*num_samples=*/1, /*replacement=*/true)
                       .squeeze(-1);

    auto log_dist = torch::log_softmax(masked, -1);
    auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    auto entropy  = -(dist * log_dist).sum(-1);

    return {action, log_prob, value, entropy};
}

// ─────────────────────────────────────────────────────────────────────────────
ActorCriticImpl::EvalResult
ActorCriticImpl::evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                          const torch::Tensor &action) {
    auto [logits, value] = forward(std::move(obs));
    const auto masked = apply_mask(logits, std::move(legal_mask));

    const auto dist     = torch::softmax(masked, -1);
    const auto log_dist = torch::log_softmax(masked, -1);
    const auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    const auto entropy  = -(dist * log_dist).sum(-1);

    return {log_prob, value, entropy};
}

} // namespace poker_ppo
