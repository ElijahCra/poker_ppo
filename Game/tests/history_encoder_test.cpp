// History-encoder variant tests: shapes, masking semantics, and
// ActorCritic integration for all HistoryEncoderKind values.

#include "config.h"
#include "network.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace {

using namespace poker_ppo;

BetHistoryConfig test_hist(HistoryEncoderKind kind) {
    BetHistoryConfig h{};
    h.enabled         = true;
    h.kind            = kind;
    h.max_history_len = 16;
    h.attn_dim        = 96;
    h.attn_heads      = 4;
    h.ffn_mult        = 3;
    h.num_blocks      = 2;
    return h;
}

constexpr int kB = 5;

// Block with n_valid tokens of deterministic junk; padded token slots are
// zero (matching ObservationBuilder::write_history, which memsets the row).
torch::Tensor make_block(const BetHistoryConfig& h, int n_valid,
                         float scale = 1.0f) {
    const int T = h.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;
    auto block = torch::zeros({kB, T * (1 + F)});
    for (int b = 0; b < kB; ++b) {
        for (int i = 0; i < n_valid; ++i) {
            block[b][i] = 1.0f;  // mask
            for (int f = 0; f < F; ++f) {
                block[b][T + i * F + f] =
                    scale * std::sin(0.7f * (b + 1) * (i + 1) * (f + 1));
            }
        }
    }
    return block;
}

template <typename Enc>
void check_shape_and_finite(Enc& enc, const BetHistoryConfig& h) {
    for (int n_valid : {0, 1, 7, h.max_history_len}) {
        torch::Tensor out = enc->forward(make_block(h, n_valid));
        ASSERT_EQ(out.dim(), 2);
        EXPECT_EQ(out.size(0), kB);
        EXPECT_EQ(out.size(1), h.attn_dim);
        EXPECT_TRUE(out.isfinite().all().item<bool>())
            << "non-finite output at n_valid=" << n_valid;
    }
}

TEST(HistoryEncoderKinds, ShapesAndFinite) {
    {
        auto h = test_hist(HistoryEncoderKind::Attention);
        HistoryEncoder e(h);
        check_shape_and_finite(e, h);
    }
    {
        auto h = test_hist(HistoryEncoderKind::Conv);
        ConvHistoryEncoder e(h);
        check_shape_and_finite(e, h);
    }
    {
        auto h = test_hist(HistoryEncoderKind::AttnPool);
        AttnPoolHistoryEncoder e(h);
        check_shape_and_finite(e, h);
    }
    {
        auto h = test_hist(HistoryEncoderKind::Flatten);
        FlattenHistoryEncoder e(h);
        check_shape_and_finite(e, h);
    }
}

// Attention variants must ignore the CONTENT of padded slots entirely
// (true masking): junk beyond the mask cannot change the output.
TEST(HistoryEncoderKinds, AttentionVariantsIgnorePaddedJunk) {
    const int n_valid = 6;

    auto run = [&](auto& enc, const BetHistoryConfig& h) {
        auto clean = make_block(h, n_valid);
        auto junk  = clean.clone();
        const int T = h.max_history_len;
        const int F = BetHistoryConfig::feat_per_action;
        // Garbage into padded token slots; mask unchanged.
        junk.narrow(1, T + n_valid * F, (T - n_valid) * F)
            .copy_(torch::randn({kB, (T - n_valid) * F}) * 50.0f);
        auto a = enc->forward(clean);
        auto b = enc->forward(junk);
        EXPECT_TRUE(torch::allclose(a, b, /*rtol=*/1e-4, /*atol=*/1e-5))
            << "padded-slot content leaked into the encoding";
    };

    {
        auto h = test_hist(HistoryEncoderKind::Attention);
        HistoryEncoder e(h);
        run(e, h);
    }
    {
        auto h = test_hist(HistoryEncoderKind::AttnPool);
        AttnPoolHistoryEncoder e(h);
        run(e, h);
    }
}

// Conv relies on zero-padded slots (k=3 bleeds neighbours), but the masked
// mean-pool must still exclude padded POSITIONS: growing T-side zero
// padding beyond the valid prefix must not change the output...
// equivalently, the encoding of n tokens must differ from the same tokens
// with the mask shortened (pooling actually respects the mask).
TEST(HistoryEncoderKinds, ConvMaskedPoolRespectsMask) {
    auto h = test_hist(HistoryEncoderKind::Conv);
    ConvHistoryEncoder e(h);

    auto full = make_block(h, 8);
    auto cut  = full.clone();
    // Same tokens, but mask the last 4 of the 8 valid positions off.
    cut.narrow(1, 4, 4).zero_();

    auto a = e->forward(full);
    auto b = e->forward(cut);
    EXPECT_FALSE(torch::allclose(a, b, /*rtol=*/1e-3, /*atol=*/1e-4))
        << "mask change did not affect the pooled encoding";
}

// Every kind must build into a full ActorCritic, forward, and survive
// clone_actor_critic (generic param copy), all selected via cfg.kind in
// one process (the env override, when unset, must not pin the kind).
TEST(HistoryEncoderKinds, ActorCriticIntegrationAndClone) {
    const auto rs  = RoundSummaryConfig{.enabled = true};
    for (auto kind : {HistoryEncoderKind::Attention,
                      HistoryEncoderKind::Conv,
                      HistoryEncoderKind::AttnPool,
                      HistoryEncoderKind::Flatten}) {
        auto hist   = test_hist(kind);
        auto layout = ObservationLayout::build(hist, rs);
        const int obs_dim = layout.total_dim;
        const int A       = 14;

        ActorCritic net(obs_dim, A, /*hidden=*/64, /*layers=*/2, hist, rs);
        auto obs  = torch::randn({kB, obs_dim});
        auto mask = torch::ones({kB, A});

        auto ar = net->get_action(obs, mask);
        EXPECT_EQ(ar.action.size(0), kB);
        EXPECT_TRUE(ar.log_prob.isfinite().all().item<bool>());

        auto copy = clone_actor_critic(net, obs_dim, A, 64, 2, hist, rs,
                                       torch::kCPU);
        auto v0 = net->get_state_value(obs, mask);
        auto v1 = copy->get_state_value(obs, mask);
        EXPECT_TRUE(torch::allclose(v0, v1, /*rtol=*/1e-5, /*atol=*/1e-6))
            << "clone diverged for kind " << static_cast<int>(kind);
    }
}

}  // namespace
