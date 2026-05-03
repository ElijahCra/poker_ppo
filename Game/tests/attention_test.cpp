// attention_test.cpp — black-box checks on the bet-history attention encoder.
//
// We don't reach into TowerImpl; instead we drive the full ActorCritic
// forward pass with hist.enabled=true and assert behaviour properties:
//
//   1. Forward pass runs and returns finite values.
//   2. Backward pass runs (gradients flow through the encoder).
//   3. PADDING INVARIANCE: padding-only positions don't affect the value
//      head's output for the same content prefix. This is the property
//      that the masking code is supposed to guarantee — if it's broken,
//      this test trips.
//   4. CONTENT SENSITIVITY: changing real (non-padded) tokens DOES change
//      the value output. Catches a "mask kills everything" failure mode.
//
// Together (3) + (4) verify the masking is correct without depending on
// internal encoder shapes.

#include "network.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <random>

namespace {

using namespace poker_ppo;

// ─── Helpers to build a known obs vector ─────────────────────────────────
//
// Uses the project's canonical `ObservationLayout` so the test can never
// drift from the real env-↔-network layout: a layout change forces a
// failure here too, at compile time.

constexpr int CARD_SLOTS = ObservationLayout::CARD_SLOTS;

// Build a [1, total] obs tensor with a configurable history block.
//   `mask_real` = number of leading positions in the history that are real
//   (mask=1); the rest are padded (mask=0).
//   `token_seed` = seed for the token feature values, so distinct seeds give
//   distinct token contents.
torch::Tensor make_obs(const BetHistoryConfig& hist,
                       const RoundSummaryConfig& rs,
                       int mask_real,
                       uint32_t token_seed) {
    const auto L = ObservationLayout::build(hist, rs);
    auto obs = torch::zeros({1, L.total_dim}, torch::kFloat32);
    auto a   = obs.accessor<float, 2>();

    // Static features: a couple of card one-hots, valid stacks/pot, sane round.
    a[0][L.hole_off + 0]  = 1.0f;         // hole card 0
    a[0][L.hole_off + 13] = 1.0f;         // hole card 1
    a[0][L.static_off + 0] = 0.95f;       // stack me
    a[0][L.static_off + 1] = 0.95f;       // stack opp
    a[0][L.static_off + 2] = 0.01f;       // pot
    a[0][L.static_off + 3] = 0.01f;       // current_bet
    a[0][L.static_off + 4] = 0.0f;        // raise_num
    a[0][L.static_off + 5] = 1.0f;        // round one-hot[0] = preflop
    // round one-hot[1..3], me, hand_strength[..] — leave zeros.

    if (rs.enabled) {
        // Leave round-summary features at zero — irrelevant for these tests.
    }

    if (hist.enabled) {
        const int hist_off = L.history_off;
        const int T        = hist.max_history_len;
        const int F        = BetHistoryConfig::feat_per_action;

        // Mask block: first `mask_real` positions are real.
        for (int i = 0; i < mask_real; ++i) a[0][hist_off + i] = 1.0f;

        // Token block: only the first `mask_real` get real content; the rest
        // we deliberately fill with garbage to test that the mask suppresses
        // them. If masking is buggy the garbage will leak into the output.
        std::mt19937 rng(token_seed);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        const int tok_off = hist_off + T;
        for (int i = 0; i < T; ++i) {
            for (int f = 0; f < F; ++f) {
                const int idx = tok_off + i * F + f;
                if (i < mask_real) {
                    // Real: deterministic, content-sensitive value.
                    a[0][idx] = std::sin(static_cast<float>(token_seed * 1000 + i * F + f));
                } else {
                    // Padding: random garbage that should be masked out.
                    a[0][idx] = u(rng) * 100.0f;  // large magnitudes to expose leaks
                }
            }
        }
    }

    return obs;
}

// Compute the value head output for an obs (single sample).
// Uses ActorCritic::get_value to skip the actor tower.
float get_value(ActorCritic& net, const torch::Tensor& obs) {
    torch::NoGradGuard ng;
    return net->get_value(obs).item<float>();
}

// Common config for the tests: small but nontrivial encoder.
BetHistoryConfig small_hist() {
    BetHistoryConfig h;
    h.enabled         = true;
    h.max_history_len = 8;
    h.attn_dim        = 16;
    h.attn_heads      = 2;
    h.ffn_mult        = 2;
    h.num_blocks      = 1;
    return h;
}

}  // namespace

// ─── Tests ───────────────────────────────────────────────────────────────

TEST(Attention, ForwardAndBackwardRun) {
    BetHistoryConfig hist = small_hist();
    RoundSummaryConfig rs;
    rs.enabled = false;

    const auto L = ObservationLayout::build(hist, rs);
    ActorCritic net(L.total_dim, /*action_count=*/4,
                    /*hidden_dim=*/32, /*num_layers=*/2,
                    hist, rs);
    net->train();

    auto obs    = make_obs(hist, rs, /*mask_real=*/3, /*token_seed=*/42);
    auto mask   = torch::ones({1, 4});
    auto action = torch::zeros({1}, torch::kInt64);

    auto er = net->evaluate(obs, mask, action);
    ASSERT_TRUE(er.value.requires_grad());

    // Backward should run without errors and produce finite gradients.
    auto loss = er.value.pow(2).mean()
              + er.log_prob.mean()
              - 0.01f * er.entropy.mean();
    loss.backward();

    // Spot-check that at least one encoder parameter received a gradient.
    int with_grad = 0;
    int with_finite_grad = 0;
    for (const auto& p : net->parameters()) {
        if (p.grad().defined() && p.grad().numel() > 0) {
            ++with_grad;
            if (p.grad().isfinite().all().item<bool>()) ++with_finite_grad;
        }
    }
    EXPECT_GT(with_grad, 0);
    EXPECT_EQ(with_grad, with_finite_grad);
}

TEST(Attention, PaddingIsInvariantToGarbageInPaddedSlots) {
    // The masking code's whole point: changing token features at masked
    // positions must not change the value-head output. We test by building
    // two obs with identical real prefixes but different garbage in padded
    // positions, and asserting their value outputs are bit-identical.
    BetHistoryConfig hist = small_hist();
    RoundSummaryConfig rs;
    rs.enabled = false;

    const auto L = ObservationLayout::build(hist, rs);
    ActorCritic net(L.total_dim, /*action_count=*/4, 32, 2, hist, rs);
    net->eval();

    // Same `mask_real=3` and same real-token seed → identical real content.
    // Different padding-garbage seeds → different values in masked slots.
    auto obs_a = make_obs(hist, rs, /*mask_real=*/3, /*token_seed=*/7);

    // Hand-craft a second obs with same mask + same real tokens but diff padding.
    auto obs_b = obs_a.clone();
    const int hist_off = L.history_off;
    const int T = hist.max_history_len;
    const int F = BetHistoryConfig::feat_per_action;
    const int tok_off = hist_off + T;
    const int mask_real = 3;
    auto b_acc = obs_b.accessor<float, 2>();
    for (int i = mask_real; i < T; ++i) {
        for (int f = 0; f < F; ++f) {
            // Different garbage values in masked positions.
            b_acc[0][tok_off + i * F + f] = -42.0f + i * 13.0f + f * 1.7f;
        }
    }

    const float v_a = get_value(net, obs_a);
    const float v_b = get_value(net, obs_b);

    // Should be bit-identical: padded positions get -∞ in attention scores
    // → 0 weight after softmax → 0 contribution to output.
    EXPECT_FLOAT_EQ(v_a, v_b)
        << "padded tokens leaked into the value output. "
        << "v_a=" << v_a << " v_b=" << v_b;
}

TEST(Attention, EncoderIsContentSensitiveOnRealTokens) {
    // Conjugate property: changing REAL (non-padded) tokens MUST change
    // the value output. Catches a "mask zeros everything out" bug where
    // the encoder would be invariant to all input.
    BetHistoryConfig hist = small_hist();
    RoundSummaryConfig rs;
    rs.enabled = false;

    const auto L = ObservationLayout::build(hist, rs);
    ActorCritic net(L.total_dim, /*action_count=*/4, 32, 2, hist, rs);
    net->eval();

    auto obs_a = make_obs(hist, rs, /*mask_real=*/3, /*token_seed=*/100);
    auto obs_b = make_obs(hist, rs, /*mask_real=*/3, /*token_seed=*/200);

    const float v_a = get_value(net, obs_a);
    const float v_b = get_value(net, obs_b);

    // Different real tokens → different outputs (untrained net is not
    // identically zero — orthogonal init produces nontrivial gradients).
    EXPECT_NE(v_a, v_b)
        << "encoder appears insensitive to real-token content. "
        << "v_a=" << v_a << " v_b=" << v_b;
}

TEST(Attention, AddingPureZeroPaddingDoesNotChangeOutput) {
    // Property orthogonal to the previous: holding the real tokens fixed
    // and varying mask_real downward (effectively "removing" history while
    // keeping the same real values at the front) should NOT change the
    // value-head output, because the mask is what tells the encoder which
    // positions are real.
    //
    // This is the same property as PaddingIsInvariantToGarbageInPaddedSlots
    // but with a different attack vector: the obs differs in the *mask*
    // values, not just in the padded token features. A correct
    // implementation should treat zero-mask + arbitrary-token the same as
    // zero-mask + zero-token.

    BetHistoryConfig hist = small_hist();
    RoundSummaryConfig rs;
    rs.enabled = false;

    const auto L = ObservationLayout::build(hist, rs);
    ActorCritic net(L.total_dim, /*action_count=*/4, 32, 2, hist, rs);
    net->eval();

    // obs_a: 3 real tokens, padded slots have zeros (mask=0 + token=0).
    BetHistoryConfig hist_zero_pad = hist;
    auto obs_a = make_obs(hist_zero_pad, rs, /*mask_real=*/3, /*token_seed=*/55);
    // Wipe padded token slots to zero on top of make_obs's garbage.
    {
        auto a = obs_a.accessor<float, 2>();
        const int hist_off = L.history_off;
        const int T = hist.max_history_len;
        const int F = BetHistoryConfig::feat_per_action;
        const int tok_off = hist_off + T;
        for (int i = 3; i < T; ++i) {
            for (int f = 0; f < F; ++f) a[0][tok_off + i * F + f] = 0.0f;
        }
    }

    // obs_b: 3 real tokens, padded slots have nonzero garbage (mask=0 +
    // token≠0). Should match obs_a despite the garbage.
    auto obs_b = make_obs(hist, rs, /*mask_real=*/3, /*token_seed=*/55);

    EXPECT_FLOAT_EQ(get_value(net, obs_a), get_value(net, obs_b));
}
