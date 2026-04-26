//  main.cpp — PPO self-play training on the Poker heads-up NLHE engine,
//  with an Elo league for progress tracking.
//
//  First arg (optional) selects a named game variant:
//      ./poker_ppo                   → full 52-card NLHE (default)
//      ./poker_ppo no_deuces_48      → drop all 2s (48-card deck)
//      ./poker_ppo short_deck_36     → drop 2s–5s (36-card short-deck)
//  Everything else (hyperparams, bet slots) is set inline below; the
//  sweep harness will replace this with a richer config loader later.

#include "league.h"
#include "metrics_logger.h"
#include "ppo.h"
#include "poker_env.h"
#include "GameConfig.hpp"

#include <chrono>
#include <iomanip>
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
    //   ./poker_ppo --strategy {serial|threadpool}  (training mode)
    //   ./poker_ppo --no-attention                  → disable bet-history encoder
    //   ./poker_ppo --round-summary                 → enable per-round summary block
    bool             benchmark_mode  = false;
    bool             scaling_mode    = false;
    bool             use_attention   = true;
    bool             use_round_summary = false;
    int              benchmark_iters = 20;
    std::vector<int> env_counts;
    std::string      variant         = "nlhe_full_52";
    std::string      strategy        = "threadpool";  // default training rollout

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
        } else if (a == "--no-attention") {
            use_attention = false;
        } else if (a == "--attention") {
            use_attention = true;
        } else if (a == "--round-summary") {
            use_round_summary = true;
        } else if (a == "--no-round-summary") {
            use_round_summary = false;
        } else {
            variant = std::string(a);
        }
    }

    if (scaling_mode && env_counts.empty()) {
        env_counts = {8, 16, 32, 64, 128};  // sensible default sweep
    }

    // ── Game variant: deck, stacks, blinds, betting structure ──────────
    PokerConfig poker_cfg;
    poker_cfg.game = makeGameConfig(variant);
    poker_cfg.seed = 0x12345;
    poker_cfg.game.validate();

    // ── BetConfig: must expose the same total action count to PPO ──────
    BetConfig bet_cfg;
    bet_cfg.num_raise_sizes    = poker_cfg.num_raise_slots();
    bet_cfg.min_raise          = 0.5;      // unused by adapter
    bet_cfg.geometric_ratio    = 1.5;      // unused by adapter
    bet_cfg.max_bets_per_round = poker_cfg.game.max_raises_per_round;

    printGame(poker_cfg.game);

    // ── PPO hyper-parameters ────────────────────────────────────────────
    PPOConfig ppo_cfg;
    ppo_cfg.total_timesteps = 200'000'000;
    ppo_cfg.num_envs        = 8;
    ppo_cfg.num_steps       = 128;
    ppo_cfg.update_epochs   = 4;
    ppo_cfg.num_minibatches = 4;
    ppo_cfg.learning_rate   = 2.5e-4f;
    ppo_cfg.ent_coef        = 0.02f;

    ppo_cfg.vf_coef         = 0.5f;
    ppo_cfg.clip_coef       = 0.2f;
    ppo_cfg.clip_vloss      = true;

    ppo_cfg.hidden_dim      = 256;
    ppo_cfg.num_layers      = 3;
    ppo_cfg.anneal_lr       = true;

    // ── Bet-history attention encoder ──────────────────────────────────
    // T = max actions per hand the encoder sees (older actions are dropped).
    // 32 comfortably covers heads-up NLHE: 4 rounds × 4 raises × 2 players
    // = 32 raise slots, plus call/check fillers — truncation is rare.
    ppo_cfg.hist.enabled         = false;   // --no-attention to disable
    ppo_cfg.hist.max_history_len = 32;
    ppo_cfg.hist.attn_dim        = 64;
    ppo_cfg.hist.attn_heads      = 4;
    ppo_cfg.hist.ffn_mult        = 4;
    ppo_cfg.hist.num_blocks      = 1;

    // ── Round-summary feature block ────────────────────────────────────
    // Cheap MLP-friendly alternative (or complement) to attention: 4 features
    // × 4 rounds appended directly to the trunk input.  Toggle independently
    // of --attention with --round-summary / --no-round-summary.
    ppo_cfg.round_summary.enabled = true;

    // Env must use the same layout so obs_dim aligns with the network split.
    poker_cfg.hist          = ppo_cfg.hist;
    poker_cfg.round_summary = ppo_cfg.round_summary;
    std::cout << "Bet-history attention encoder: "
              << (ppo_cfg.hist.enabled ? "ON" : "OFF") << "\n"
              << "Round-summary block          : "
              << (ppo_cfg.round_summary.enabled ? "ON" : "OFF") << "\n";

    // ── Trainer ─────────────────────────────────────────────────────────
    // CPU beats CUDA for this config (3x256 MLP @ batch=32): kernel-launch
    // overhead dominates compute on such a small network. Revisit if you
    // scale the network up (hidden_dim ≥ 512) or num_envs (≥ 128).
    //torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::mps::is_available() ? torch::kMPS : torch::kCPU;
    torch::Device device = torch::kCPU;
    std::cout << "Using device: "<<device<<"\n";

    PokerEnvironmentFactory factory(poker_cfg);

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
        std::cout << "\n[benchmark mode] comparing serial / threadpool\n";
        trainer.benchmark_rollouts(benchmark_iters, /*warmup=*/3);
        return 0;
    }

    // Pick the rollout strategy used by train().
    if (strategy == "serial") {
        trainer.set_rollout_fn([](PPOTrainer& t) { t.collect_rollout_serial(); });
    } else if (strategy == "threadpool") {
        trainer.set_rollout_fn([](PPOTrainer& t) { t.collect_rollout_threadpool(); });
    } else {
        std::cerr << "unknown --strategy '" << strategy
                  << "' (expected serial|threadpool)\n";
        return 1;
    }
    std::cout << "Rollout strategy: " << strategy << "\n";

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
    league_cfg.num_hands_per_match = 1000;
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

    // Five default anchors: uniform, random_init, always_call, always_raise,
    // pair_caller.  See league.h for the policy of each.
    league.add_default_anchors();
    std::cout << "League anchors:";
    for (const auto& a : league.anchors()) std::cout << "  " << a->name();
    std::cout << "\n";

    // How often to evaluate against the anchors (synchronous so it doesn't
    // contend with rollout/update wall-clock).
    constexpr int snapshot_every = 200;   // updates

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

        // Synchronous evaluation every `snapshot_every` updates.  Synchronous
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
    });

    std::cout << "\nStarting training for "
              << ppo_cfg.total_timesteps << " steps...\n\n";

    trainer.train();

    // Final evaluation on the trained network.
    std::cout << "\nFinal league evaluation...\n";
    auto final_results = league.evaluate(trainer.network());
    metrics.log_league(/*update=*/-1, /*step=*/-1, final_results);
    league.print_results(final_results);

    const std::string model_path = "poker_ppo_model_" + poker_cfg.game.name + ".pt";
    trainer.save(model_path);
    std::cout << "\nModel saved to " << model_path << "\n";
    return 0;
}
