//  main.cpp — PPO self-play training on the QFR heads-up NLHE engine,
//  with an Elo league for progress tracking.
//
//  First arg (optional) selects a named game variant:
//      ./poker_ppo                   → full 52-card NLHE (default)
//      ./poker_ppo no_deuces_48      → drop all 2s (48-card deck)
//      ./poker_ppo short_deck_36     → drop 2s–5s (36-card short-deck)
//  Everything else (hyperparams, bet slots) is set inline below; the
//  sweep harness will replace this with a richer config loader later.

#include "elo_league.h"
#include "metrics_logger.h"
#include "ppo.h"
#include "qfr_env.h"
#include "GameConfig.hpp"

#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
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

    // Tiny MLP (3×256) at batch ≈ 32 does not benefit from Torch's intra-op
    // pool — the sync overhead is net-negative, and it steals cores from
    // the rollout worker threads. Force single-threaded so workers and the
    // inference processor each get a dedicated core.
    //at::set_num_threads(1);
    //at::set_num_interop_threads(1);

    // ── CLI parse ──────────────────────────────────────────────────────
    // Recognised forms:
    //   ./poker_ppo                                    → train, default variant
    //   ./poker_ppo <variant>                          → train, named variant
    //   ./poker_ppo --benchmark [iters]                → A/B rollout bench
    //   ./poker_ppo <variant> --benchmark [iters]
    //   ./poker_ppo --scaling N1,N2,N3,... [iters]    → scaling sweep
    //   ./poker_ppo <variant> --scaling 8,16,32 [iters]
    //   ./poker_ppo --strategy {serial|coroutine|threadpool}  (training mode)
    bool             benchmark_mode  = false;
    bool             scaling_mode    = false;
    int              benchmark_iters = 20;
    std::vector<int> env_counts;
    std::string      variant         = "nlhe_full_52";
    std::string      strategy        = "coroutine";  // default training rollout

    auto parse_int_list = [](std::string_view s) {
        std::vector<int> out;
        size_t start = 0;
        while (start <= s.size()) {
            size_t end = s.find(',', start);
            if (end == std::string_view::npos) end = s.size();
            if (end > start) {
                try { out.push_back(std::stoi(std::string(s.substr(start, end - start)))); }
                catch (...) {}
            }
            start = end + 1;
        }
        return out;
    };

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--benchmark") {
            benchmark_mode = true;
            if (i + 1 < argc) {
                try { benchmark_iters = std::stoi(argv[i + 1]); ++i; }
                catch (...) { /* leave default */ }
            }
        } else if (a == "--scaling") {
            scaling_mode = true;
            if (i + 1 < argc) {
                env_counts = parse_int_list(argv[i + 1]);
                ++i;
            }
            if (i + 1 < argc) {
                try { benchmark_iters = std::stoi(argv[i + 1]); ++i; }
                catch (...) { /* leave default */ }
            }
        } else if (a == "--strategy") {
            if (i + 1 < argc) { strategy = argv[i + 1]; ++i; }
        } else {
            variant = std::string(a);
        }
    }

    if (scaling_mode && env_counts.empty()) {
        env_counts = {8, 16, 32, 64, 128};  // sensible default sweep
    }

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
    ppo_cfg.total_timesteps = 200'000'000;
    ppo_cfg.num_envs        = 64;
    ppo_cfg.num_steps       = 128;
    ppo_cfg.update_epochs   = 8;         // more critic passes per rollout
    ppo_cfg.num_minibatches = 4;
    ppo_cfg.learning_rate   = 2.0e-4f;
    ppo_cfg.ent_coef        = 0.01f;     // was 0.05 — too large for a 6-action IIG with
                                         // near-zero advantages; pinned policy ~uniform
    ppo_cfg.vf_coef         = 1.0f;      // critic gets a real share of trunk gradient
    ppo_cfg.clip_coef       = 0.2f;      // standard PPO; gives policy room (was 0.1)
    ppo_cfg.clip_vloss      = false;     // CRITICAL: with reward scale ≫ clip, clipped vloss
                                         // becomes constant outside V_old±ε → zero critic gradient
    // Rewards come from the env in mbb (±100k for a full-stack pot). Scale by
    // 10× big_blind so terminal returns land roughly in [-10, 10] and typical
    // hands around ±1-3 — well-conditioned for a small MLP. Scaling by the
    // full stack drove returns to ~0.03 avg, putting the critic below its
    // numerical noise floor (EV went strongly negative, advantages turned to
    // garbage, policy learned in the wrong direction). Elo / league reporting
    // still reads raw env reward, so bb/hand display is unchanged.
    ppo_cfg.reward_scale    = 1.0f / (10.0f * static_cast<float>(qfr_cfg.game.big_blind));
    ppo_cfg.hidden_dim      = 256;
    ppo_cfg.num_layers      = 3;
    ppo_cfg.anneal_lr       = true;

    // ── Trainer ─────────────────────────────────────────────────────────
    // CPU beats CUDA for this config (3x256 MLP @ batch=32): kernel-launch
    // overhead dominates compute on such a small network. Revisit if you
    // scale the network up (hidden_dim ≥ 512) or num_envs (≥ 128).
    torch::Device device = torch::kCUDA;
    std::cout << "Using device: CPU\n";

    QFRPokerEnvironmentFactory factory(qfr_cfg);

    if (scaling_mode) {
        std::cout << "\n[scaling mode] sweep over num_envs={";
        for (size_t i = 0; i < env_counts.size(); ++i) {
            std::cout << env_counts[i] << (i + 1 < env_counts.size() ? "," : "");
        }
        std::cout << "}, iters=" << benchmark_iters << "\n\n";
        scaling_benchmark(factory, bet_cfg, ppo_cfg, device,
                          env_counts, benchmark_iters, /*warmup=*/2);
        return 0;
    }

    PPOTrainer trainer(factory, bet_cfg, ppo_cfg, device);

    if (benchmark_mode) {
        std::cout << "\n[benchmark mode] comparing serial / coroutine / threadpool\n";
        trainer.benchmark_rollouts(benchmark_iters, /*warmup=*/3);
        return 0;
    }

    // Pick the rollout strategy used by train().
    if (strategy == "serial") {
        trainer.set_rollout_fn([](PPOTrainer& t) { t.collect_rollout_serial(); });
    } else if (strategy == "threadpool") {
        trainer.set_rollout_fn([](PPOTrainer& t) { t.collect_rollout_threadpool(); });
    } else if (strategy == "coroutine") {
        trainer.set_rollout_fn([](PPOTrainer& t) { t.collect_rollout_coroutine(); });
    } else {
        std::cerr << "unknown --strategy '" << strategy
                  << "' (expected serial|coroutine|threadpool)\n";
        return 1;
    }
    std::cout << "Rollout strategy: " << strategy << "\n";

    // ── Elo league ──────────────────────────────────────────────────────
    // Query obs_dim and action_count from a throw-away env instance so the
    // league's snapshot networks match the trainer's network architecture.
    int obs_dim;
    int action_count;
    {
        auto tmp     = factory.create(bet_cfg);
        obs_dim      = tmp->obs_dim();
        action_count = tmp->bet_config().action_count();
    }

    EloLeague::Config elo_cfg;
    elo_cfg.num_hands_per_match = 1000;
    elo_cfg.num_parallel_envs   = 32;
    elo_cfg.k_factor            = 32.0f;
    elo_cfg.initial_rating      = 1200.0f;
    elo_cfg.max_checkpoints     = 15;

    EloLeague league(factory, bet_cfg,
                     obs_dim,
                     action_count,
                     ppo_cfg.hidden_dim,
                     ppo_cfg.num_layers,
                     elo_cfg,
                     device);

    // Anchor the untrained network at 1200 — every later rating reads as
    // "how much stronger than the initial random network".
    league.add_checkpoint(trainer.network(), 0, 0, "initial");
    league.set_anchor(0, 1200.0f);

    // Second anchor: a *true* uniform-over-legal-actions baseline.  Built by
    // zeroing the actor head so logits are constant → softmax over the legal
    // mask is uniform.  Guards against "1200 → 1200 looks healthy because
    // both checkpoints are equally bad" — any real progress should beat both.
    {
        ActorCritic uniform(obs_dim, action_count,
                            ppo_cfg.hidden_dim, ppo_cfg.num_layers);
        torch::NoGradGuard ng;
        for (auto& kv : uniform->named_parameters()) {
            if (kv.key().find("actor_head") != std::string::npos) {
                kv.value().zero_();
            }
        }
        const int uniform_idx =
            league.add_checkpoint(uniform, 0, 0, "uniform");
        league.set_anchor(uniform_idx, 1200.0f);
    }

    // How often to snapshot a checkpoint and play it against the pool.
    constexpr int snapshot_every = 200;   // updates

    // Action labels for the histogram printout.
    auto action_label = [&](int a) -> std::string {
        if (a == 0) return "F";
        if (a == 1) return "C";
        const int raise_idx = a - 2;
        const int n_pot     = static_cast<int>(qfr_cfg.game.pot_fractions.size());
        if (raise_idx < n_pot) return "R" + std::to_string(raise_idx);
        return "AI";
    };

    // ── Metrics logging ─────────────────────────────────────────────────
    // CSVs land under runs/<timestamp>/ and are tailed by tools/plot_live.py.
    MetricsLogger metrics(make_run_dir());
    std::cout << "Metrics dir: " << metrics.run_dir() << "\n"
              << "  (live plots: `python tools/plot_live.py "
              << metrics.run_dir() << "`)\n";

    // Elo evaluation runs in a background thread so it never stalls training.
    // We hold at most one evaluation future at a time; the next snapshot waits
    // for it only if the previous one hasn't finished yet (which is rare given
    // snapshot_every=200 updates and evaluations taking a few seconds).
    std::future<void> elo_future;
    std::mutex        print_mutex;  // keep training + elo output from interleaving

    trainer.set_log_callback([&](const PPOTrainer::UpdateStats& s) {
        // Per-update CSV row (cheap; cb fires once per update).
        metrics.log_update(s);

        if (s.update % 10 == 0) {
            std::lock_guard<std::mutex> lk(print_mutex);
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

        // Snapshot and evaluate periodically.
        if (s.update > 0 && s.update % snapshot_every == 0) {
            // Wait for any still-running evaluation before touching the league.
            if (elo_future.valid()) elo_future.wait();

            // add_checkpoint is a deep copy — safe to do on the training thread.
            const std::string label = "u" + std::to_string(s.update);
            const int latest_idx = league.add_checkpoint(
                trainer.network(), s.update, s.global_step, label);

            // Fire the rest off in the background — training resumes immediately.
            elo_future = std::async(std::launch::async,
                [&, latest_idx, update = s.update, step = s.global_step]() {
                    league.evaluate_latest();

                    auto vs_initial = league.play_match(latest_idx, 0);
                    auto vs_uniform = league.play_match(latest_idx, 1);

                    // Persist Elo + match summary to runs/<...>/elo.csv.
                    const float latest_rating =
                        league.checkpoints()[latest_idx].rating;
                    metrics.log_elo(update, step, latest_rating,
                                    vs_initial.win_rate_a,
                                    vs_initial.avg_reward_a * 10.0f,
                                    vs_uniform.win_rate_a,
                                    vs_uniform.avg_reward_a * 10.0f);

                    // Mode-collapse diagnostic: action histogram from uniform match.
                    int64_t total = 0;
                    for (auto c : vs_uniform.action_counts_a) total += c;

                    std::lock_guard<std::mutex> lk(print_mutex);
                    std::cout << "\n[Elo standings after update " << update << "]\n";
                    league.print_standings();
                    std::cout << std::fixed << std::setprecision(3)
                              << "  vs initial:  win=" << vs_initial.win_rate_a
                              << "  bb/hand=" << (vs_initial.avg_reward_a * 10.0f) << "\n"
                              << "  vs uniform:  win=" << vs_uniform.win_rate_a
                              << "  bb/hand=" << (vs_uniform.avg_reward_a * 10.0f) << "\n";
                    std::cout << "  action mix (vs uniform):";
                    if (total > 0) {
                        std::cout << std::setprecision(1);
                        for (size_t a = 0; a < vs_uniform.action_counts_a.size(); ++a) {
                            const float pct = 100.0f
                                * static_cast<float>(vs_uniform.action_counts_a[a])
                                / static_cast<float>(total);
                            std::cout << "  " << action_label(static_cast<int>(a))
                                      << "=" << pct << "%";
                        }
                    }
                    std::cout.unsetf(std::ios::fixed);
                    std::cout << "\n\n";
                });
        }
    });

    std::cout << "\nStarting training for "
              << ppo_cfg.total_timesteps << " steps...\n\n";

    trainer.train();

    // Drain any background Elo evaluation before the final tournament.
    if (elo_future.valid()) elo_future.wait();

    // Final full round-robin for a sharper read-out at the end.
    std::cout << "\nRunning final round-robin tournament...\n";
    league.run_tournament();
    std::cout << "\n[Final Elo league standings]\n";
    league.print_standings();

    const std::string model_path = "poker_ppo_model_" + qfr_cfg.game.name + ".pt";
    trainer.save(model_path);
    std::cout << "\nModel saved to " << model_path << "\n";
    return 0;
}
