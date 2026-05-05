#include "network.h"
#include "features.h"

#include <torch/torch.h>
#include <stdexcept>

namespace poker_ppo {

TowerImpl::TowerImpl(int obs_dim, int output_dim,
                     int hidden_dim, int num_layers,
                     float head_init_std,
                     BetHistoryConfig    hist,
                     RoundSummaryConfig  round_summary)
    : hist_(hist),
      round_summary_(round_summary),
      layout_(ObservationLayout::build(hist, round_summary))
{
    if (layout_.total_dim != obs_dim) {
        throw std::invalid_argument(
            "Tower: obs_dim does not match ObservationLayout::build(hist, round_summary)");
    }
    const int tower_static_dim = layout_.static_off + ObservationLayout::FEAT_STATIC;
    int trunk_in_dim = tower_static_dim + layout_.round_summary_dim;

    // Whole block dead-code-eliminated when the build flag is off.
    if constexpr (features::ATTENTION_ENCODER) {
        if (hist_.enabled) {
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

            // CLS embedding (D) feeds the trunk alongside round_summary if both on.
            trunk_in_dim += D;
        }
    }

    torch::nn::Sequential trunk;
    int in_dim = trunk_in_dim;
    for (int i = 0; i < num_layers; ++i) {
        trunk->push_back(torch::nn::Linear(in_dim, hidden_dim));
        trunk->push_back(torch::nn::Tanh());
        in_dim = hidden_dim;
    }
    trunk_ = register_module("trunk", trunk);
    head_  = register_module("head",  torch::nn::Linear(hidden_dim, output_dim));

    // Orthogonal for trunk/head, xavier for attention.
    for (auto& m : trunk_->modules(/*include_self=*/false)) {
        if (auto* lin = m->as<torch::nn::Linear>()) {
            torch::nn::init::orthogonal_(lin->weight, std::sqrt(2.0));
            torch::nn::init::constant_(lin->bias, 0.0);
        }
    }
    torch::nn::init::orthogonal_(head_->weight, head_init_std);
    torch::nn::init::constant_(head_->bias, 0.0);

    if constexpr (features::ATTENTION_ENCODER) {
        if (hist_.enabled) {
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
    }
}

torch::Tensor
TowerImpl::encode_history(const torch::Tensor& history_block) {
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

    auto add_mask = (1.0 - attn_mask).unsqueeze(1).unsqueeze(1) * kAttentionMaskLogit;

    const int64_t L = T + 1;
    for (int b = 0; b < hist_.num_blocks; ++b) {
        auto h = attn_ln_[b]->forward(x);
        auto qkv = qkv_proj_[b]->forward(h);          // [B, L, 3D]
        auto qkv_split = qkv.chunk(3, /*dim=*/-1);
        auto reshape_heads = [&](torch::Tensor t) {
            return t.reshape({B, L, H, d_head}).transpose(1, 2).contiguous();
        };
        auto q = reshape_heads(qkv_split[0]);
        auto k = reshape_heads(qkv_split[1]);
        auto v = reshape_heads(qkv_split[2]);

        auto scores = torch::matmul(q, k.transpose(-2, -1))
                    / std::sqrt(static_cast<double>(d_head));
        scores = scores + add_mask;
        auto attn = torch::softmax(scores, /*dim=*/-1);

        auto out = torch::matmul(attn, v);
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

torch::Tensor TowerImpl::forward(torch::Tensor obs) {
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

    if constexpr (features::ATTENTION_ENCODER) {
        if (hist_.enabled) {
            auto history_blk = obs.narrow(1, layout_.history_off,
                                          layout_.history_dim);
            parts.push_back(encode_history(history_blk));
        }
    }

    torch::Tensor trunk_in = (parts.size() == 1)
        ? parts[0]
        : torch::cat(parts, /*dim=*/-1);
    auto features = trunk_->forward(trunk_in);
    return head_->forward(features);
}

ActorCriticImpl::ActorCriticImpl(int obs_dim, int action_count,
                                 int hidden_dim, int num_layers,
                                 BetHistoryConfig    hist,
                                 RoundSummaryConfig  round_summary)
{
    actor_  = register_module("actor",
                              Tower(obs_dim, action_count,
                                    hidden_dim, num_layers,
                                    /*head_init_std=*/0.01f,
                                    hist, round_summary));
    critic_ = register_module("critic",
                              Tower(obs_dim, /*output_dim=*/1,
                                    hidden_dim, num_layers,
                                    /*head_init_std=*/1.0f,
                                    hist, round_summary));
}

std::pair<torch::Tensor, torch::Tensor>
ActorCriticImpl::forward(torch::Tensor obs) {
    auto logits = actor_->forward(obs);
    auto value  = critic_->forward(obs).squeeze(-1);
    return {logits, value};
}

torch::Tensor ActorCriticImpl::get_value(torch::Tensor obs) {
    return critic_->forward(obs).squeeze(-1);
}

torch::Tensor
ActorCriticImpl::apply_mask(torch::Tensor logits, torch::Tensor mask) {
    return logits + (1.0f - mask) * kIllegalActionLogit;
}

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

ActorCriticImpl::EvalResult
ActorCriticImpl::evaluate(torch::Tensor obs, torch::Tensor legal_mask,
                          const torch::Tensor &action) {
    auto [logits, value] = forward(obs);
    const auto masked = apply_mask(logits, legal_mask);

    const auto dist     = torch::softmax(masked, -1);
    const auto log_dist = torch::log_softmax(masked, -1);
    const auto log_prob = log_dist.gather(-1, action.unsqueeze(-1)).squeeze(-1);
    const auto entropy  = -(dist * log_dist).sum(-1);

    return {log_prob, log_dist, value, entropy};
}

torch::Tensor
ActorCriticImpl::masked_log_probs(torch::Tensor obs, torch::Tensor legal_mask) {
    // Actor-only forward — the critic's value head isn't needed for the
    // KL term, so saving the second tower call is essentially free.
    auto logits       = actor_->forward(obs);
    const auto masked = apply_mask(logits, legal_mask);
    return torch::log_softmax(masked, -1);
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

    torch::NoGradGuard ng;
    auto sp = src->parameters();
    auto dp = dst->parameters();
    TORCH_CHECK(sp.size() == dp.size(),
                "clone_actor_critic: parameter count mismatch");
    for (size_t i = 0; i < sp.size(); ++i) {
        dp[i].copy_(sp[i].detach().to(device));
    }
    auto sb = src->buffers();
    auto db = dst->buffers();
    TORCH_CHECK(sb.size() == db.size(),
                "clone_actor_critic: buffer count mismatch");
    for (size_t i = 0; i < sb.size(); ++i) {
        db[i].copy_(sb[i].detach().to(device));
    }
    dst->eval();
    return dst;
}

} // namespace poker_ppo
