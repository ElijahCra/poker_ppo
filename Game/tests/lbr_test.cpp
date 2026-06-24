// LBR integration smoke: the evaluator must run end-to-end against a real
// ActorCritic + PokerEnvironment and return a finite bb/hand with a sane
// win rate. Numeric correctness of the equity core lives in
// range_equity_test; this guards the range/Bayes/decision loop wiring.

#include "config.h"
#include "lbr.h"
#include "network.h"
#include "poker_env.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

namespace {

using namespace poker_ppo;

TEST(LBR, RunsEndToEndAndReturnsFiniteBound) {
    torch::manual_seed(20260624);
    PokerConfig pc = kPokerConfig;
    PokerEnvironmentFactory factory(pc);

    LBRConfig cfg;
    cfg.num_hands         = 60;     // smoke — keep it quick
    cfg.equity_mc_samples = 200;
    cfg.seed              = 42;

    LBREvaluator lbr(factory, config::kBetConfig, cfg, torch::kCPU);

    // A freshly-initialised target (near-uniform policy) is a valid
    // opponent to attack — we only assert the loop produces sane output.
    int obs_dim, action_count;
    {
        auto tmp     = factory.create(config::kBetConfig);
        obs_dim      = tmp->obs_dim();
        action_count = tmp->bet_config().action_count();
    }
    ActorCritic target(obs_dim, action_count,
                       config::kPPOConfig.hidden_dim,
                       config::kPPOConfig.num_layers,
                       config::kPPOConfig.hist,
                       config::kPPOConfig.round_summary);
    target->eval();

    auto r = lbr.evaluate(target);

    EXPECT_EQ(r.num_hands, cfg.num_hands);
    EXPECT_TRUE(std::isfinite(r.bb_per_hand));
    EXPECT_TRUE(std::isfinite(r.mbb_per_hand));
    EXPECT_GE(r.lbr_win_rate, 0.0);
    EXPECT_LE(r.lbr_win_rate, 1.0);
    // mbb/hand and bb/hand must agree up to the big-blind scale.
    EXPECT_NEAR(r.bb_per_hand,
                r.mbb_per_hand / pc.game.big_blind, 1e-6);
}

}  // namespace
