// ForeachAdam / foreach_clip_grad_norm equivalence tests.
//
// The trainer swapped torch::optim::Adam + torch::nn::utils::clip_grad_norm_
// for the multi-tensor versions in optim.h purely for launch-overhead
// reasons; these tests pin down that the math is unchanged by training two
// identically-initialised MLPs side by side and demanding the parameters
// stay (numerically) identical.

#include "optim.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

namespace {

using poker_ppo::ForeachAdam;
using poker_ppo::foreach_clip_grad_norm;

torch::nn::Sequential make_mlp() {
    return torch::nn::Sequential(
        torch::nn::Linear(12, 32),
        torch::nn::Tanh(),
        torch::nn::Linear(32, 32),
        torch::nn::Tanh(),
        torch::nn::Linear(32, 4));
}

void copy_params(torch::nn::Sequential& dst, const torch::nn::Sequential& src) {
    torch::NoGradGuard ng;
    auto sp = src->parameters();
    auto dp = dst->parameters();
    ASSERT_EQ(sp.size(), dp.size());
    for (size_t i = 0; i < sp.size(); ++i) dp[i].copy_(sp[i]);
}

// One synthetic regression step on `net`; same data on both nets per step.
torch::Tensor step_loss(torch::nn::Sequential& net,
                        const torch::Tensor& x, const torch::Tensor& y) {
    return (net->forward(x) - y).pow(2).mean();
}

// Train both optimisers in lockstep for `steps`; assert params match.
void run_lockstep(double lr, double eps, double max_norm, int steps) {
    torch::manual_seed(7);
    auto ref_net = make_mlp();
    auto fe_net  = make_mlp();
    copy_params(fe_net, ref_net);

    torch::optim::Adam ref_opt(
        ref_net->parameters(), torch::optim::AdamOptions(lr).eps(eps));
    ForeachAdam fe_opt(fe_net->parameters(), lr, 0.9, 0.999, eps);

    for (int s = 0; s < steps; ++s) {
        auto x = torch::randn({16, 12});
        auto y = torch::randn({16, 4});

        ref_opt.zero_grad();
        step_loss(ref_net, x, y).backward();
        if (max_norm > 0) {
            torch::nn::utils::clip_grad_norm_(ref_net->parameters(), max_norm);
        }
        ref_opt.step();

        fe_opt.zero_grad();
        step_loss(fe_net, x, y).backward();
        if (max_norm > 0) {
            foreach_clip_grad_norm(fe_opt.params(), max_norm);
        }
        fe_opt.step();
    }

    auto rp = ref_net->parameters();
    auto fp = fe_net->parameters();
    for (size_t i = 0; i < rp.size(); ++i) {
        EXPECT_TRUE(torch::allclose(rp[i], fp[i], /*rtol=*/1e-5, /*atol=*/1e-7))
            << "param " << i << " diverged; max |Δ| = "
            << (rp[i] - fp[i]).abs().max().item<float>();
    }
}

TEST(ForeachAdam, MatchesTorchAdamNoClip) {
    run_lockstep(/*lr=*/3e-3, /*eps=*/1e-8, /*max_norm=*/0.0, /*steps=*/25);
}

TEST(ForeachAdam, MatchesTorchAdamWithActiveClip) {
    // max_norm small enough that the clip engages nearly every step.
    run_lockstep(/*lr=*/3e-3, /*eps=*/1e-8, /*max_norm=*/0.05, /*steps=*/25);
}

TEST(ForeachAdam, MatchesTorchAdamCustomEps) {
    run_lockstep(/*lr=*/1e-3, /*eps=*/1e-5, /*max_norm=*/0.5, /*steps=*/25);
}

TEST(ForeachClipGradNorm, MatchesReferenceNorm) {
    torch::manual_seed(11);
    auto net = make_mlp();
    auto x = torch::randn({8, 12});
    auto y = torch::randn({8, 4});
    step_loss(net, x, y).backward();

    // Reference norm before any clipping.
    double ref_sq = 0.0;
    for (const auto& p : net->parameters()) {
        ref_sq += p.grad().norm(2).pow(2).item<double>();
    }
    const double ref_norm = std::sqrt(ref_sq);

    auto total = foreach_clip_grad_norm(net->parameters(), /*max_norm=*/0.1);
    EXPECT_NEAR(total.item<double>(), ref_norm, 1e-5);

    // Post-clip norm must be ≈ max_norm (clip was active).
    double post_sq = 0.0;
    for (const auto& p : net->parameters()) {
        post_sq += p.grad().norm(2).pow(2).item<double>();
    }
    EXPECT_NEAR(std::sqrt(post_sq), 0.1, 1e-4);
}

TEST(ForeachClipGradNorm, NoOpWhenUnderMaxNorm) {
    torch::manual_seed(13);
    auto net = make_mlp();
    auto x = torch::randn({8, 12});
    auto y = torch::randn({8, 4});
    step_loss(net, x, y).backward();

    std::vector<torch::Tensor> before;
    for (const auto& p : net->parameters()) before.push_back(p.grad().clone());

    foreach_clip_grad_norm(net->parameters(), /*max_norm=*/1e9);

    auto params = net->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        // clamp_max(1.0) path multiplies by exactly 1.0 → bitwise no-op.
        EXPECT_TRUE(torch::equal(params[i].grad(), before[i]))
            << "grad " << i << " changed despite norm < max_norm";
    }
}

}  // namespace
