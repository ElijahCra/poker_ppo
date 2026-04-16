//  main.cpp  — example showing how to wire up the PPO trainer
//
//  Replace StubPokerEnvironment with your real Texas Hold'em implementation.

#include "ppo.h"
#include <iostream>
#include <random>

using namespace poker_ppo;

// ═════════════════════════════════════════════════════════════════════════════
// STUB environment — replace this with your real implementation
// ═════════════════════════════════════════════════════════════════════════════

class StubPokerEnvironment : public IPokerEnvironment {
public:
    explicit StubPokerEnvironment(const BetConfig& cfg)
        : cfg_(cfg), rng_(std::random_device{}())
    {}

    int obs_dim() const override {
        // Example: 52 card bits + pot + stack + street + …
        // Choose whatever encoding you like.
        return 128;
    }

    const BetConfig& bet_config() const override { return cfg_; }

    StepResult reset() override {
        step_count_    = 0;
        done_          = false;
        player_        = 0;
        bets_this_round_ = 0;
        return make_result(0.0f, false);
    }

    StepResult step(int action) override {
        (void)action;
        step_count_++;

        // Toggle player
        player_ = 1 - player_;

        if (Action::is_raise(action))
            bets_this_round_++;

        // Dummy termination after some steps
        if (step_count_ >= 8) {
            done_ = true;
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            return make_result(dist(rng_), true);
        }
        return make_result(0.0f, false);
    }

    int          current_player()    const override { return player_; }
    torch::Tensor observation()      const override {
        return torch::randn({obs_dim()});
    }
    torch::Tensor legal_action_mask() const override {
        auto mask = torch::ones({cfg_.action_count()});
        // Mask out raises if bet cap reached
        if (bets_this_round_ >= cfg_.max_bets_per_round) {
            for (int i = 0; i < cfg_.num_raise_sizes; ++i)
                mask[Action::Raise(i)] = 0.0f;
        }
        return mask;
    }
    bool is_terminal() const override { return done_; }

private:
    StepResult make_result(float reward, bool done) {
        return { observation(), reward, done, legal_action_mask() };
    }

    BetConfig cfg_;
    std::mt19937 rng_;
    int  step_count_ = 0;
    int  player_     = 0;
    bool done_       = false;
    int  bets_this_round_ = 0;
};

// ─────────────────────────────────────────────────────────────────────────────

class StubFactory : public IPokerEnvironmentFactory {
public:
    std::unique_ptr<IPokerEnvironment> create(const BetConfig& cfg) override {
        return std::make_unique<StubPokerEnvironment>(cfg);
    }
};

// ═════════════════════════════════════════════════════════════════════════════
// main
// ═════════════════════════════════════════════════════════════════════════════

int main() {
    // ── configure bet sizes ──────────────────────────────────────────────
    BetConfig bet_cfg;
    bet_cfg.num_raise_sizes    = 5;       // 5 raise options
    bet_cfg.min_raise          = 0.5;     // 0.5× pot
    bet_cfg.geometric_ratio    = 2.0;     // 0.5, 1, 2, 4, 8 × pot
    bet_cfg.max_bets_per_round = 3;       // up to 3 raises per player per round

    std::cout << "Action space (" << bet_cfg.action_count() << " actions):\n"
              << "  [0] Fold\n"
              << "  [1] Check/Call\n";
    for (int i = 0; i < bet_cfg.num_raise_sizes; ++i) {
        std::cout << "  [" << Action::Raise(i) << "] Raise "
                  << bet_cfg.raise_amount(i) << "x\n";
    }

    // ── configure PPO ────────────────────────────────────────────────────
    PPOConfig ppo_cfg;
    ppo_cfg.total_timesteps = 1'000'000;
    ppo_cfg.num_envs        = 16;
    ppo_cfg.num_steps       = 128;
    ppo_cfg.update_epochs   = 4;
    ppo_cfg.num_minibatches = 4;
    ppo_cfg.ent_coef        = 0.05f;   // higher entropy for IIGs
    ppo_cfg.learning_rate   = 2.5e-4f;
    ppo_cfg.hidden_dim      = 512;
    ppo_cfg.num_layers      = 3;

    // ── train ────────────────────────────────────────────────────────────
    StubFactory factory;
    PPOTrainer trainer(factory, bet_cfg, ppo_cfg);

    trainer.set_log_callback([](const PPOTrainer::UpdateStats& s) {
        if (s.update % 50 == 0) {
            std::cout << "update=" << s.update
                      << "  step=" << s.global_step
                      << "  pg_loss=" << s.policy_loss
                      << "  vf_loss=" << s.value_loss
                      << "  entropy=" << s.entropy
                      << "  kl=" << s.approx_kl
                      << "  clip=" << s.clip_fraction
                      << "  ev=" << s.explained_variance
                      << "  lr=" << s.learning_rate
                      << "\n";
        }
    });

    std::cout << "\nStarting training for "
              << ppo_cfg.total_timesteps << " steps...\n\n";

    trainer.train();

    // ── save ─────────────────────────────────────────────────────────────
    trainer.save("poker_ppo_model.pt");
    std::cout << "\nModel saved to poker_ppo_model.pt\n";

    return 0;
}
