// vrpo_advantage_test.cpp — unit tests for RolloutBuffer::compute_returns,
// the unified GAE / Expected-SARSA(λ) "Q-boosting" trace.
//
// Two guarantees:
//   1. When value==vbar (the V-critic path), the trace must reduce EXACTLY
//      to Generalized Advantage Estimation — checked against an independent
//      GAE reference.
//   2. When value≠vbar (the VRPO Q-critic path), it must match the
//      Expected-SARSA(λ) formula Â = Q_target − V̄, Q_target = Q + Σ(λγ)^k δ⁺,
//      δ⁺ = r + γV̄(s') − Q(s,a).

#include "rollout.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <utility>
#include <vector>

using namespace poker_ppo;

namespace {

struct Traj {
    std::vector<float> r, value, vbar;   // per-step reward, Q/V, V̄/V
};

// Single player-0/env-0 trajectory through the real buffer; terminal tail
// (bootstrap V̄_next = 0). Returns {advantages, returns}.
std::pair<std::vector<float>, std::vector<float>>
run_buffer(const Traj& tj, float gamma, float lam) {
    const int T = static_cast<int>(tj.r.size());
    const int obs_dim = 4, A = 3;
    RolloutBuffer buf(T, /*num_envs=*/1, obs_dim, A, torch::kCPU);

    auto obs  = torch::zeros({obs_dim});
    auto mask = torch::ones({A});
    for (int t = 0; t < T; ++t)
        buf.push(/*player=*/0, /*env=*/0, obs, /*action=*/0, /*log_prob=*/0.0f,
                 tj.r[t], /*done=*/0.0f, tj.value[t], tj.vbar[t], mask);

    auto bv = torch::zeros({2, 1});   // unused: tail is terminal
    auto bt = torch::ones({2, 1});    // terminal → V̄_next = 0
    buf.compute_returns(gamma, lam, bv, bt);

    auto batch = buf.flatten();
    auto adv = batch.advantages.to(torch::kCPU).contiguous();
    auto ret = batch.returns.to(torch::kCPU).contiguous();
    std::vector<float> a(T), r(T);
    for (int t = 0; t < T; ++t) { a[t] = adv[t].item<float>(); r[t] = ret[t].item<float>(); }
    return {a, r};
}

// Independent GAE reference (valid when value plays the role of V(s)).
std::pair<std::vector<float>, std::vector<float>>
gae_reference(const Traj& tj, float gamma, float lam) {
    const int T = static_cast<int>(tj.r.size());
    std::vector<float> adv(T), ret(T);
    float last = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float next_v  = (t == T - 1) ? 0.0f : tj.value[t + 1];
        const float nonterm = (t == T - 1) ? 0.0f : 1.0f;
        const float delta   = tj.r[t] + gamma * next_v * nonterm - tj.value[t];
        last    = delta + gamma * lam * nonterm * last;
        adv[t]  = last;
        ret[t]  = last + tj.value[t];
    }
    return {adv, ret};
}

// Independent Expected-SARSA(λ) reference.
std::pair<std::vector<float>, std::vector<float>>
expected_sarsa_reference(const Traj& tj, float gamma, float lam) {
    const int T = static_cast<int>(tj.r.size());
    std::vector<float> adv(T), ret(T);
    float trace = 0.0f;
    for (int t = T - 1; t >= 0; --t) {
        const float next_vbar = (t == T - 1) ? 0.0f : tj.vbar[t + 1];
        const float nonterm   = (t == T - 1) ? 0.0f : 1.0f;
        const float delta     = tj.r[t] + gamma * next_vbar * nonterm - tj.value[t];
        trace   = delta + gamma * lam * nonterm * trace;
        ret[t]  = tj.value[t] + trace;     // Q_target
        adv[t]  = ret[t] - tj.vbar[t];     // Â = Q_target − V̄
    }
    return {adv, ret};
}

void expect_close(const std::vector<float>& a, const std::vector<float>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 1e-5f);
}

}  // namespace

TEST(VrpoAdvantage, ReducesToGaeWhenValueEqualsVbar) {
    Traj tj;
    tj.r     = {0.0f, 0.0f, 1.0f, -0.5f};
    tj.value = {0.2f, 0.1f, 0.4f,  0.3f};
    tj.vbar  = tj.value;                    // V-critic path
    const float gamma = 0.99f, lam = 0.9f;

    auto [adv, ret] = run_buffer(tj, gamma, lam);
    auto [gadv, gret] = gae_reference(tj, gamma, lam);
    expect_close(adv, gadv);
    expect_close(ret, gret);
}

TEST(VrpoAdvantage, MatchesExpectedSarsaWhenValueDiffersFromVbar) {
    Traj tj;
    tj.r     = {0.0f,  0.0f,  0.0f, 2.0f};
    tj.value = {0.5f,  0.3f, -0.1f, 0.9f};   // Q(s,a)
    tj.vbar  = {0.2f,  0.1f,  0.0f, 0.4f};   // V̄(s) = Σ π Q
    const float gamma = 1.0f, lam = 0.95f;

    auto [adv, ret] = run_buffer(tj, gamma, lam);
    auto [eadv, eret] = expected_sarsa_reference(tj, gamma, lam);
    expect_close(adv, eadv);
    expect_close(ret, eret);
}

TEST(VrpoAdvantage, AdvantageEqualsReturnMinusVbar) {
    Traj tj;
    tj.r     = {1.0f, -1.0f, 0.5f};
    tj.value = {0.1f,  0.2f, 0.3f};
    tj.vbar  = {0.05f, 0.15f, 0.25f};
    auto [adv, ret] = run_buffer(tj, 0.97f, 0.5f);
    for (size_t t = 0; t < adv.size(); ++t)
        EXPECT_NEAR(adv[t], ret[t] - tj.vbar[t], 1e-5f);
}
