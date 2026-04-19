//  main.cpp — PPO self-play training on the QFR heads-up NLHE engine.
//
//  First arg (optional) selects a named game variant:
//      ./poker_ppo                   → full 52-card NLHE (default)
//      ./poker_ppo no_deuces_48      → drop all 2s (48-card deck)
//      ./poker_ppo short_deck_36     → drop 2s–5s (36-card short-deck)
//  Everything else (hyperparams, bet slots) is set inline below; the
//  sweep harness will replace this with a richer config loader later.

#include "ppo.h"
#include "qfr_env.h"
#include "GameConfig.hpp"

#include <iostream>
#include <string>
#include <string_view>

using namespace poker_ppo;

namespace {

::Game::GameConfig makeGameConfig(std::string_view name) {
    if (name == "no_deuces_48") return ::Game::GameConfig::noDeuces();
    if (name == "short_deck_36") return ::Game::GameConfig::shortDeck36();

    // Default: full 52-card NLHE with pot-fraction raise slots.
    ::Game::GameConfig cfg;
    cfg.name                 = "nlhe_full_52";
    cfg.initial_stack        = 100'000;
    cfg.small_blind          = 500;
    cfg.big_blind            = 1000;
    cfg.min_bet              = 1000;
    cfg.min_raise            = 1000;
    cfg.max_raises_per_round = 4;
    cfg.pot_fractions        = {0.5, 1.0, 2.0};
    cfg.include_all_in_slot  = true;
    return cfg;
}

void printGame(const ::Game::GameConfig& g) {
    std::cout << "─── Game variant: " << g.name << " ──────────────────\n"
              << "  deck cards      : " << int(g.deck_size())
              << "  (excluded " << g.excluded_cards.size() << "/52)\n"
              << "  initial stack   : " << g.initial_stack << " mbb\n"
              << "  blinds          : " << g.small_blind << " / " << g.big_blind << " mbb\n"
              << "  min bet / raise : " << g.min_bet << " / " << g.min_raise << " mbb\n"
              << "  max raises/rnd  : " << int(g.max_raises_per_round) << "\n"
              << "  pot fractions   : {";
    for (size_t i = 0; i < g.pot_fractions.size(); ++i) {
        std::cout << g.pot_fractions[i]
                  << (i + 1 < g.pot_fractions.size() ? ", " : "");
    }
    std::cout << "}  all-in slot: " << (g.include_all_in_slot ? "yes" : "no") << "\n";

    std::cout << "Action space (" << g.action_count() << " actions):\n"
              << "  [0] Fold\n  [1] Check/Call\n";
    for (size_t i = 0; i < g.pot_fractions.size(); ++i) {
        std::cout << "  [" << (2 + i) << "] Raise " << g.pot_fractions[i] << "× pot\n";
    }
    if (g.include_all_in_slot) {
        std::cout << "  [" << (2 + g.pot_fractions.size()) << "] All-in\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);  // flush stdout as it arrives (for PTY/log capture)
    const std::string variant = (argc > 1) ? argv[1] : "nlhe_full_52";

    // ── Game variant: deck, stacks, blinds, betting structure ──────────
    QFRConfig qfr_cfg;
    qfr_cfg.game = makeGameConfig(variant);
    qfr_cfg.seed = 0x12345;
    qfr_cfg.game.validate();

    // ── BetConfig: must expose the same total action count to PPO ──────
    BetConfig bet_cfg;
    bet_cfg.num_raise_sizes    = qfr_cfg.num_raise_slots();
    bet_cfg.min_raise          = 1.0;      // unused by adapter
    bet_cfg.geometric_ratio    = 2.0;      // unused by adapter
    bet_cfg.max_bets_per_round = qfr_cfg.game.max_raises_per_round;

    printGame(qfr_cfg.game);

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

    const std::string model_path = "poker_ppo_model_" + qfr_cfg.game.name + ".pt";
    trainer.save(model_path);
    std::cout << "\nModel saved to " << model_path << "\n";
    return 0;
}
