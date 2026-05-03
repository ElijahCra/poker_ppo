#include "commands.h"

#include "best_response.h"
#include "league.h"
#include "metrics_logger.h"
#include "poker_env.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

namespace poker_ppo {

int cmd_train(IPokerEnvironmentFactory& factory,
              const PokerConfig&        poker_cfg,
              torch::Device             device,
              PPOTrainer::Strategy      strategy)
{
    PPOTrainer trainer(factory, device);
    trainer.set_rollout_strategy(strategy);

    // Local references to the compile-time configs so the call-sites read
    // naturally and we don't repeat the `config::` qualification.
    const PPOConfig& ppo_cfg = config::kPPOConfig;
    const BetConfig& bet_cfg = config::kBetConfig;

    // ── League (anchor-relative bb/hand tracking) ──────────────────────
    // Query obs_dim and action_count from a throw-away env instance so the
    // league's helper networks match the trainer's network architecture.
    int obs_dim;
    int action_count;
    {
        auto tmp     = factory.create(bet_cfg);
        obs_dim      = tmp->obs_dim();
        action_count = tmp->bet_config().action_count();
    }

    League::Config league_cfg;
    league_cfg.num_hands_per_match = 10000;
    league_cfg.num_parallel_envs   = 32;
    // reward_norm in poker_env is 10 * big_blind; one scaled-reward unit thus
    // corresponds to 10 BB.  Keep this in sync if you change the env's scale.
    league_cfg.bb_per_unit_reward  = 10.0f;

    League league(factory, bet_cfg,
                  obs_dim,
                  action_count,
                  ppo_cfg.hidden_dim,
                  ppo_cfg.num_layers,
                  ppo_cfg.hist,
                  ppo_cfg.round_summary,
                  league_cfg,
                  device);

    league.add_default_anchors();
    std::cout << "League anchors:";
    for (const auto& a : league.anchors()) std::cout << "  " << a->name();
    std::cout << "\n";

    // How often to evaluate against the anchors (synchronous so it doesn't
    // contend with rollout/update wall-clock).
    constexpr int snapshot_every = 200;   // updates

    // ── Approximate best-response evaluator ─────────────────────────────
    // Train a fresh PPO exploiter against a frozen snapshot of the trained
    // network for a lower bound on its exploitability. Toggleable via
    // BestResponseConfig::enabled. See include/best_response.h for details.
    const BestResponseConfig& br_cfg = config::kBRConfig;

    std::unique_ptr<BestResponseEvaluator> br_eval;
    if (br_cfg.enabled) {
        br_eval = std::make_unique<BestResponseEvaluator>(
            factory, bet_cfg,
            obs_dim, action_count,
            ppo_cfg.hidden_dim, ppo_cfg.num_layers,
            ppo_cfg.hist, ppo_cfg.round_summary,
            br_cfg, device);
        std::cout << "Best-response evaluator: ON  (every "
                  << br_cfg.eval_every << " updates, "
                  << br_cfg.num_exploiter_seeds << " seeds × "
                  << br_cfg.updates_per_eval << " exploiter updates, "
                  << br_cfg.eval_hands << "-hand eval match)\n";
    } else {
        std::cout << "Best-response evaluator: OFF\n";
    }

    // Action labels for the histogram printout.
    auto action_label = [&](int a) -> std::string {
        if (a == 0) return "F";
        if (a == 1) return "C";
        const int raise_idx = a - 2;
        const int n_pot     = static_cast<int>(poker_cfg.game.pot_fractions.size());
        if (raise_idx < n_pot) return "R" + std::to_string(raise_idx);
        return "AI";
    };

    // ── Metrics logging ─────────────────────────────────────────────────
    // CSVs land under runs/<timestamp>/ and are tailed by tools/plot_live.py.
    MetricsLogger metrics(make_run_dir());
    std::cout << "Metrics dir: " << metrics.run_dir() << "\n"
              << "  (live plots: `python tools/plot_live.py "
              << metrics.run_dir() << "`)\n";

    trainer.set_log_callback([&](const PPOTrainer::UpdateStats& s) {
        // Per-update CSV row (cheap; cb fires once per update).
        metrics.log_update(s);

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

        // Synchronous evaluation every `snapshot_every` updates. Synchronous
        // (rather than async) so the league's wall-clock cost is *not* added
        // to the next rollout/update timing — it shows up only as the gap
        // between consecutive [update K] log lines.
        if (s.update > 0 && s.update % snapshot_every == 0) {
            using clock = std::chrono::steady_clock;
            using ms    = std::chrono::duration<double, std::milli>;

            auto t_eval0 = clock::now();
            auto results = league.evaluate(trainer.network());
            const double eval_ms = ms(clock::now() - t_eval0).count();

            metrics.log_league(s.update, s.global_step, results);

            std::cout << "\n[league eval @ update " << s.update
                      << "  duration=" << eval_ms << "ms]\n";
            league.print_results(results);

            // Action mix (vs uniform anchor — first registered) for visual
            // mode-collapse check.
            if (!results.empty()) {
                int uniform_idx = -1;
                for (size_t i = 0; i < results.size(); ++i) {
                    if (results[i].anchor_name == "uniform") {
                        uniform_idx = static_cast<int>(i);
                        break;
                    }
                }
                if (uniform_idx >= 0) {
                    const auto& vs_u = results[uniform_idx];
                    int64_t total = 0;
                    for (auto c : vs_u.action_counts_a) total += c;
                    if (total > 0) {
                        std::cout << "  action mix (vs uniform):"
                                  << std::fixed << std::setprecision(1);
                        for (size_t a = 0; a < vs_u.action_counts_a.size(); ++a) {
                            const float pct = 100.0f
                                * static_cast<float>(vs_u.action_counts_a[a])
                                / static_cast<float>(total);
                            std::cout << "  " << action_label(static_cast<int>(a))
                                      << "=" << pct << "%";
                        }
                        std::cout.unsetf(std::ios::fixed);
                        std::cout << "\n";
                    }
                }
            }
            std::cout << "\n";
        }

        // Approximate best-response evaluation. Synchronous so its wall
        // time shows up only as a gap between consecutive [update K] lines,
        // rather than being counted toward the next rollout/update timing.
        if (br_eval && s.update > 0 && s.update % br_cfg.eval_every == 0) {
            auto br_result = br_eval->evaluate(
                trainer.network(), s.update, s.global_step);
            metrics.log_best_response(br_result);

            std::cout << "[best-response eval @ update " << s.update
                      << "  seeds=" << br_result.num_seeds
                      << "  bb/hand max=" << std::fixed << std::setprecision(3)
                      << br_result.bb_per_hand_a
                      << "  mean=" << br_result.bb_per_hand_mean
                      << "  min=" << br_result.bb_per_hand_min
                      << "  std=" << br_result.bb_per_hand_std
                      << "  best-win%=" << std::setprecision(1)
                      << (100.0f * br_result.win_rate_a)
                      << "  duration=" << std::setprecision(0)
                      << br_result.wall_ms << "ms]"
                      << std::defaultfloat << "\n\n";
        }
    });

    std::cout << "\nStarting training for "
              << ppo_cfg.total_timesteps << " steps...\n\n";

    trainer.train();

    // Final evaluation on the trained network.
    std::cout << "\nFinal league evaluation...\n";
    auto final_results = league.evaluate(trainer.network());
    metrics.log_league(/*update=*/-1, /*step=*/-1, final_results);
    league.print_results(final_results);

    // poker_cfg.game.name is now a std::string_view (constexpr-friendly);
    // bridge to std::string explicitly for the path concatenation.
    const std::string model_path =
        std::string("poker_ppo_model_") + std::string(poker_cfg.game.name) + ".pt";
    trainer.save(model_path);
    std::cout << "\nModel saved to " << model_path << "\n";
    return 0;
}

} // namespace poker_ppo
