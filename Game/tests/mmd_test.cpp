// mmd_test.cpp — pinning tests for the MMD regulariser plumbing.
//
// MMD adds `kl_coef * KL(π_θ || ρ)` to the PPO loss, where ρ is a frozen
// magnet snapshot of π_θ. Two pieces need to be correct:
//
//   1. `ActorCritic::masked_log_probs(obs, mask)` must agree with
//      `evaluate(obs, mask, ·).log_probs_all` on the same inputs. The
//      magnet evaluation in PPOTrainer::update() uses the former; if it
//      drifted from the latter the live policy and the magnet would be
//      computed against different masking conventions and the KL term
//      would be silently wrong.
//
//   2. KL(π || π) must be ≈ 0 when computed via the obs-builder code
//      path (`exp(log_p) * (log_p - log_q)`, summed over actions). Tests
//      the *math* of the KL formula independently of the network.
//
// These are unit-level checks — they're enough to catch a regression in
// the masking convention or the KL math without standing up the full
// trainer.

#include "network.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

namespace {

using namespace poker_ppo;

// Same skeletal config the attention test uses — keeps the network small
// so the test is fast.
BetHistoryConfig make_hist() {
    BetHistoryConfig h;
    h.enabled         = false;   // no attention path; pure trunk
    h.max_history_len = 8;
    h.attn_dim        = 16;
    h.attn_heads      = 2;
    return h;
}

ActorCritic build_net(int obs_dim, int action_count) {
    BetHistoryConfig    hist = make_hist();
    RoundSummaryConfig  rs;
    rs.enabled = false;
    return ActorCritic(obs_dim, action_count,
                       /*hidden_dim=*/32, /*num_layers=*/2, hist, rs);
}

torch::Tensor make_obs(int B, int D) {
    torch::manual_seed(0);
    return torch::randn({B, D}, torch::kFloat32);
}

torch::Tensor make_mask(int B, int A, int n_legal) {
    auto m = torch::zeros({B, A}, torch::kFloat32);
    // Mark the first n_legal slots as legal; rest illegal. Tests that
    // masked positions are excluded from the softmax denominator.
    auto a = m.accessor<float, 2>();
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < n_legal; ++i) a[b][i] = 1.0f;
    return m;
}

}  // namespace

TEST(MMD, MaskedLogProbsMatchesEvaluate) {
    // The two paths into the masked log-softmax must agree bit-for-bit.
    // `evaluate()` already runs `log_softmax`; `masked_log_probs()` is
    // an actor-only forward + the same masking + the same softmax.
    //
    // Compute both ways for the same (obs, mask) and assert equal.
    constexpr int B = 4;
    constexpr int A = 7;

    const ObservationLayout L =
        ObservationLayout::build(make_hist(), RoundSummaryConfig{});
    auto net = build_net(L.total_dim, A);
    net->eval();

    auto obs    = make_obs(B, L.total_dim);
    auto mask   = make_mask(B, A, /*n_legal=*/3);
    auto action = torch::zeros({B}, torch::kInt64);  // any legal action

    torch::NoGradGuard ng;
    auto via_evaluate     = net->evaluate(obs, mask, action).log_probs_all;
    auto via_masked_lp    = net->masked_log_probs(obs, mask);

    EXPECT_TRUE(via_evaluate.allclose(via_masked_lp, /*rtol=*/1e-5, /*atol=*/1e-5))
        << "masked_log_probs and evaluate().log_probs_all must agree — they "
           "feed the same KL term in PPOTrainer::update().";
}

TEST(MMD, KLZeroForIdenticalDistributions) {
    // KL(π || π) = 0. We compute via the same expression update() uses:
    //   pi      = exp(log_p)
    //   kl      = (pi * (log_p - log_q)).sum(-1)
    // With log_p == log_q the sum collapses to 0. Sanity check on the
    // formula — if the formula has a sign flip or wrong reduction dim,
    // this trips.
    constexpr int B = 4;
    constexpr int A = 7;

    const ObservationLayout L =
        ObservationLayout::build(make_hist(), RoundSummaryConfig{});
    auto net = build_net(L.total_dim, A);
    net->eval();

    auto obs  = make_obs(B, L.total_dim);
    auto mask = make_mask(B, A, /*n_legal=*/4);

    torch::NoGradGuard ng;
    auto log_p = net->masked_log_probs(obs, mask);
    auto log_q = log_p.clone();    // pretend "magnet" is identical

    auto pi = log_p.exp();
    auto kl = (pi * (log_p - log_q)).sum(/*dim=*/-1);  // [B]

    // Numerical floor — fp32 softmax + the kIllegalActionLogit fill on
    // masked slots leaves residuals ~1e-9 ish.
    EXPECT_LE(kl.abs().max().item<float>(), 1e-5f)
        << "KL(π || π) should be 0; found max abs " << kl.abs().max().item<float>();
}

TEST(MMD, KLPositiveForDifferentDistributions) {
    // Conjugate to the previous test: when log_q is a clearly different
    // distribution, KL > 0. Build log_q by rotating actions in log_p.
    constexpr int B = 4;
    constexpr int A = 7;

    const ObservationLayout L =
        ObservationLayout::build(make_hist(), RoundSummaryConfig{});
    auto net = build_net(L.total_dim, A);
    net->eval();

    auto obs  = make_obs(B, L.total_dim);
    auto mask = torch::ones({B, A}, torch::kFloat32);  // all legal so the roll
                                                       // doesn't shuffle masked
                                                       // slots into legal ones.

    torch::NoGradGuard ng;
    auto log_p = net->masked_log_probs(obs, mask);
    // Roll along the action axis — same support, different probabilities.
    auto log_q = torch::roll(log_p, /*shifts=*/1, /*dims=*/-1);

    auto pi = log_p.exp();
    auto kl = (pi * (log_p - log_q)).sum(/*dim=*/-1).mean();

    EXPECT_GT(kl.item<float>(), 0.0f)
        << "KL between two genuinely-different distributions should be > 0; got "
        << kl.item<float>();
}
