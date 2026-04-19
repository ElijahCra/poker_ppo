//  main.cpp — PPO self-play training on the QFR heads-up NLHE engine.

#include "ppo.h"
#include "qfr_env.h"

#include <iostream>

using namespace poker_ppo;

int main() {
    // ── QFR-side config: pot-fraction raise slots + all-in ──────────────
    QFRConfig qfr_cfg;
    qfr_cfg.pot_fractions        = {0.5, 1.0, 2.0};
    qfr_cfg.include_all_in_slot  = true;
    qfr_cfg.min_bet              = 1000;
    qfr_cfg.min_raise            = 1000;
    qfr_cfg.max_raises_per_round = 4;
    qfr_cfg.seed                 = 0x12345;

    // ── BetConfig: must expose the same total action count to PPO ──────
    BetConfig bet_cfg;
    bet_cfg.num_raise_sizes    = qfr_cfg.num_raise_slots();  // fractions + all-in
    bet_cfg.min_raise          = 1.0;                        // unused by adapter
    bet_cfg.geometric_ratio    = 2.0;                        // unused by adapter
    bet_cfg.max_bets_per_round = qfr_cfg.max_raises_per_round;

    std::cout << "Action space (" << bet_cfg.action_count() << " actions):\n"
              << "  [0] Fold\n"
              << "  [1] Check/Call\n";
    for (size_t i = 0; i < qfr_cfg.pot_fractions.size(); ++i) {
        std::cout << "  [" << (2 + i) << "] Raise " << qfr_cfg.pot_fractions[i]
                  << "× pot\n";
    }
    if (qfr_cfg.include_all_in_slot) {
        std::cout << "  [" << (2 + qfr_cfg.pot_fractions.size()) << "] All-in\n";
    }

    // ── PPO hyper-parameters ────────────────────────────────────────────
    PPOConfig ppo_cfg;
    ppo_cfg.total_timesteps = 2'000'000;
    ppo_cfg.num_envs        = 32;
    ppo_cfg.num_steps       = 128;
    ppo_cfg.update_epochs   = 4;
    ppo_cfg.num_minibatches = 4;
    ppo_cfg.learning_rate   = 2.5e-4f;
    ppo_cfg.ent_coef        = 0.02f;
    ppo_cfg.hidden_dim      = 256;
    ppo_cfg.num_layers      = 3;

    // ── Trainer ─────────────────────────────────────────────────────────
    QFRPokerEnvironmentFactory factory(qfr_cfg);
    PPOTrainer trainer(factory, bet_cfg, ppo_cfg);

    trainer.set_log_callback([](const PPOTrainer::UpdateStats& s) {
        if (s.update % 10 == 0) {
            std::cout << "update=" << s.update
                      << "  step="   << s.global_step
                      << "  pg="     << s.policy_loss
                      << "  vf="     << s.value_loss
                      << "  H="      << s.entropy
                      << "  kl="     << s.approx_kl
                      << "  clip="   << s.clip_fraction
                      << "  ev="     << s.explained_variance
                      << "  lr="     << s.learning_rate
                      << "\n";
        }
    });

    std::cout << "\nStarting training for "
              << ppo_cfg.total_timesteps << " steps...\n\n";

    trainer.train();

    trainer.save("poker_ppo_model.pt");
    std::cout << "\nModel saved to poker_ppo_model.pt\n";
    return 0;
}
