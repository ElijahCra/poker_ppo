//  main.cpp — PPO self-play training on the Poker heads-up NLHE engine,
//  with an Elo league for progress tracking.
//
//  Game variant + hyperparameters are baked in via `include/config.h`
//  (`kPPOConfig`, `kBetConfig`, `kBRConfig`, `kPokerConfig`). Edit those
//  and rebuild to change the run.

#include "best_response.h"
#include "league.h"
#include "metrics_logger.h"
#include "ppo.h"
#include "poker_env.h"
#include "GameConfig.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

using namespace poker_ppo;

namespace {

void printGame(const ::Game::DefaultGameConfig& g) {
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

// ─────────────────────────────────────────────────────────────────────────────
// run_play — interactive REPL exposing one PokerEnvironment + the trained
// network over stdin/stdout. Driven by tools/play.py for the UI.
//
// Protocol: each command from the client is a single line. The server
// responds with multiple `key value...` lines terminated by an `OK` line.
// All amounts are in mbb; cards are integer ids (rank<<2 | suit) in [0, 52).
//
// Commands:
//   INFO                — emit static config (action_count, obs_dim, blinds,
//                         pot_fractions, etc.).
//   STATE               — emit current game state (player, cards, stacks,
//                         pot, mask, obs, done, utility).
//   STEP <action_idx>   — apply the given action; emit new STATE.
//   RESET               — reset to a fresh hand; emit new STATE.
//   MODEL               — run the trained network on the current observation;
//                         emit chosen action + full softmax probs + value.
//   QUIT                — exit cleanly.
//
// Errors: emit a single `ERR <message>` line followed by `OK`.
// ─────────────────────────────────────────────────────────────────────────────

void run_play(IPokerEnvironmentFactory& factory,
              const BetConfig&    bet_cfg,
              torch::Device       device,
              const std::string&  model_path) {
    PPOTrainer trainer(factory, device);
    trainer.load(model_path);
    auto& net = trainer.network();
    net->eval();

    // Single env for play. We need the concrete PokerEnvironment for the
    // state-inspection accessors (hole_cards, pot, etc.).
    auto env_base = factory.create(bet_cfg);
    auto* env = dynamic_cast<PokerEnvironment*>(env_base.get());
    if (!env) {
        std::cerr << "ERR factory did not produce PokerEnvironment\n";
        return;
    }
    env->reset();

    auto emit_info = [&]() {
        const auto& g = env->game_config();
        std::cout << "obs_dim "      << env->obs_dim()             << "\n";
        std::cout << "action_count " << bet_cfg.action_count()     << "\n";
        std::cout << "initial_stack " << g.initial_stack           << "\n";
        std::cout << "small_blind "   << g.small_blind             << "\n";
        std::cout << "big_blind "     << g.big_blind               << "\n";
        std::cout << "min_raise "     << g.min_raise               << "\n";
        std::cout << "max_raises_per_round " << int(g.max_raises_per_round) << "\n";
        std::cout << "pot_fractions " << g.pot_fractions.size();
        for (double f : g.pot_fractions) std::cout << " " << f;
        std::cout << "\n";
        std::cout << "has_allin " << (g.include_all_in_slot ? 1 : 0) << "\n";
        std::cout << "OK" << std::endl;
    };

    auto emit_state = [&]() {
        std::cout << "cur_player " << env->current_player() << "\n";
        std::cout << "round "      << env->round()          << "\n";
        std::cout << "pot "        << env->pot()            << "\n";
        std::cout << "cb "         << env->current_bet()    << "\n";
        std::cout << "raises "     << env->raise_num()      << "\n";
        const auto h0 = env->hole_cards(0);
        const auto h1 = env->hole_cards(1);
        std::cout << "hole_p0 " << h0[0] << " " << h0[1] << "\n";
        std::cout << "hole_p1 " << h1[0] << " " << h1[1] << "\n";
        std::cout << "stacks "  << env->stack(0) << " " << env->stack(1) << "\n";
        const int n_comm = env->community_count();
        std::cout << "board " << n_comm;
        for (int i = 0; i < n_comm; ++i)
            std::cout << " " << env->community_card(i);
        std::cout << "\n";
        const auto mask = env->legal_action_mask();
        const auto m_acc = mask.accessor<float, 1>();
        std::cout << "mask " << mask.size(0);
        for (int i = 0; i < mask.size(0); ++i)
            std::cout << " " << (m_acc[i] > 0.5f ? 1 : 0);
        std::cout << "\n";
        const auto obs = env->observation();
        const auto o_acc = obs.accessor<float, 1>();
        std::cout << "obs " << obs.size(0);
        for (int i = 0; i < obs.size(0); ++i)
            std::cout << " " << o_acc[i];
        std::cout << "\n";
        std::cout << "done " << (env->is_terminal() ? 1 : 0) << "\n";
        if (env->is_terminal()) {
            std::cout << "utility_p0 " << env->terminal_utility(0) << "\n";
            std::cout << "utility_p1 " << env->terminal_utility(1) << "\n";
        }
        std::cout << "OK" << std::endl;
    };

    auto emit_model = [&]() {
        torch::NoGradGuard ng;
        auto obs  = env->observation().unsqueeze(0).to(device);
        auto mask = env->legal_action_mask().unsqueeze(0).to(device);
        // Pool conditioning at deploy time: leave is_pool unset → defaults
        // to self-play (class 0) inside value_with_offset. Actor's logits
        // don't depend on it anyway.
        auto [logits, value] = net->forward(obs);
        const auto masked = logits + (1.0f - mask) * (-1e8f);
        const auto probs  = torch::softmax(masked, -1).squeeze(0).to(torch::kCPU).contiguous();
        // Sample one action stochastically (matches training rollout policy).
        const auto ar = net->get_action(obs, mask);
        const auto sampled = ar.action.to(torch::kCPU).item<int64_t>();
        // Greedy (argmax) action for diagnostics.
        const auto greedy = std::get<1>(probs.max(0)).item<int64_t>();
        std::cout << "sampled "  << sampled << "\n";
        std::cout << "greedy "   << greedy  << "\n";
        std::cout << "value "    << value.squeeze(0).item<float>() << "\n";
        const auto p_acc = probs.accessor<float, 1>();
        std::cout << "probs " << probs.size(0);
        for (int i = 0; i < probs.size(0); ++i)
            std::cout << " " << p_acc[i];
        std::cout << "\n";
        std::cout << "OK" << std::endl;
    };

    // Ready signal so the client can sync.
    std::cout << "READY" << std::endl;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        try {
            if (cmd == "INFO") {
                emit_info();
            } else if (cmd == "STATE") {
                emit_state();
            } else if (cmd == "STEP") {
                int action;
                if (!(iss >> action)) {
                    std::cout << "ERR bad_step_arg\nOK" << std::endl;
                    continue;
                }
                env->step(action);
                emit_state();
            } else if (cmd == "RESET") {
                env->reset();
                emit_state();
            } else if (cmd == "MODEL") {
                emit_model();
            } else if (cmd == "QUIT") {
                break;
            } else {
                std::cout << "ERR unknown_cmd\nOK" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "ERR " << e.what() << "\nOK" << std::endl;
        }
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
    //   ./poker_ppo                                  → train (default config)
    //   ./poker_ppo --benchmark [iters]              → A/B rollout bench
    //   ./poker_ppo --play <model_path>              → interactive REPL
    //   ./poker_ppo --strategy {serial|threadpool}   → training rollout strategy
    bool             benchmark_mode  = false;
    bool             play_mode       = false;
    std::string      play_model_path;
    int              benchmark_iters = 20;
    std::string      strategy        = "threadpool";  // default training rollout

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--benchmark") {
            benchmark_mode = true;
            if (i + 1 < argc) {
                try { benchmark_iters = std::stoi(argv[i + 1]); ++i; }
                catch (...) { /* leave default */ }
            }
        } else if (a == "--play") {
            play_mode = true;
            if (i + 1 < argc) { play_model_path = argv[i + 1]; ++i; }
        } else if (a == "--strategy") {
            if (i + 1 < argc) { strategy = argv[i + 1]; ++i; }
        } else {
            std::cerr << "unknown argument '" << a << "'\n";
            return 1;
        }
    }

    // ── Game variant: deck, stacks, blinds, betting structure ──────────
    // `kPokerConfig` is the single compile-time PokerConfig consumed
    // throughout the binary. Edit `include/config.h` + `Game/GameConfig.hpp`
    // to change variants.
    PokerConfig poker_cfg = kPokerConfig;
    poker_cfg.game.validate();

    // ── BetConfig / PPOConfig: pulled directly from `include/config.h` ──
    // The trainer already references `config::kPPOConfig` / `kBetConfig`
    // internally; the local copies below just exist so the existing
    // call-sites (printing, league, run_play) keep their `const PPOConfig&`
    // / `const BetConfig&` parameter shape. Edit `include/config.h` to
    // change values.
    BetConfig bet_cfg = config::kBetConfig;
    PPOConfig ppo_cfg = config::kPPOConfig;

    printGame(poker_cfg.game);
    std::cout << "Bet-history attention encoder: "
              << (ppo_cfg.hist.enabled ? "ON" : "OFF") << "\n"
              << "Round-summary block          : "
              << (ppo_cfg.round_summary.enabled ? "ON" : "OFF") << "\n"
              << "Opponent pool                : "
              << (ppo_cfg.opp_pool.enabled ? "ON" : "OFF");
    if (ppo_cfg.opp_pool.enabled) {
        std::cout << "  (size=" << ppo_cfg.opp_pool.max_size
                  << ", snapshot_every=" << ppo_cfg.opp_pool.snapshot_every
                  << ", warmup=" << ppo_cfg.opp_pool.warmup_updates
                  << ", p_use_pool=" << ppo_cfg.opp_pool.p_use_pool
                  << ", max_unique=" << ppo_cfg.opp_pool.max_unique_per_rollout << ")";
    }
    std::cout << "\n";

    // ── Trainer ─────────────────────────────────────────────────────────
    // CPU beats CUDA for this config (3x256 MLP @ batch=32): kernel-launch
    // overhead dominates compute on such a small network. Revisit if you
    // scale the network up (hidden_dim ≥ 512) or num_envs (≥ 128).
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    std::cout << "Using device: "<<device<<"\n";

    PokerEnvironmentFactory factory(poker_cfg);

    if (play_mode) {
        if (play_model_path.empty()) {
            std::cerr << "--play requires a model path: ./poker_ppo --play <path>\n";
            return 1;
        }
        std::cerr << "[play] loading " << play_model_path << "\n";
        run_play(factory, bet_cfg, device, play_model_path);
        return 0;
    }

    PPOTrainer trainer(factory, device);

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

    // Five default anchors: uniform, random_init, always_call, always_raise,
    // pair_caller.  See league.h for the policy of each.
    league.add_default_anchors();
    std::cout << "League anchors:";
    for (const auto& a : league.anchors()) std::cout << "  " << a->name();
    std::cout << "\n";

    // How often to evaluate against the anchors (synchronous so it doesn't
    // contend with rollout/update wall-clock).
    constexpr int snapshot_every = 200;   // updates

    // ── Approximate best-response evaluator (Timbers et al. 2020) ──────
    // Train a fresh PPO exploiter against a frozen snapshot of the trained
    // network to get a lower bound on its exploitability. Toggleable via
    // BestResponseConfig::enabled. See include/best_response.h for details.
    BestResponseConfig br_cfg;
    br_cfg.enabled            = true;
    br_cfg.eval_every         = 1000;     // updates between evaluations
    // Each eval trains a fresh exploiter for `updates_per_eval` updates.
    // 200 was too short — most evals just measured "can an undertrained
    // exploiter beat the target" (always no). 1000 gives enough budget for
    // the exploiter to actually find weaknesses; ~5× the per-eval wall
    // cost vs 200, still cheap relative to 1000 main updates between evals.
    br_cfg.updates_per_eval   = 1000;
    br_cfg.num_envs           = 32;
    br_cfg.num_steps          = 128;
    br_cfg.update_epochs      = 4;
    br_cfg.num_minibatches    = 4;
    br_cfg.learning_rate      = 3.0e-4f;
    br_cfg.ent_coef           = 0.01f;
    // warm_start=false: each eval starts the exploiter from random init, so
    // the curve is a time-stationary bound. warm_start=true gives a tighter
    // bound but conflates "the policy got more exploitable" with "the
    // exploiter has had more compute." See br.csv for the resulting curve.
    br_cfg.warm_start         = false;
    // Post-training eval-only match — the bb/hand reported as the BR estimate
    // comes from this match, not from rewards collected during exploiter
    // training (which would bias the bound downward).
    br_cfg.eval_hands         = 5000;
    // Per-eval, train this many independent fresh exploiters and report the
    // max-bb/hand as the canonical bound. Each individual exploiter's
    // bb/hand is a valid lower bound; the max across seeds is the tightest
    // bound the budget can produce, and the curve is much less noisy than
    // any single seed. Mean/min/std are also recorded for diagnostics.
    br_cfg.num_exploiter_seeds = 3;
    br_cfg.bb_per_unit_reward = league_cfg.bb_per_unit_reward;
    br_cfg.seed               = 0xCAFEBABE;

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
