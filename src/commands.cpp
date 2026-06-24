#include "commands.h"

#include "best_response.h"
#include "lbr.h"
#include "league.h"
#include "metrics_logger.h"
#include "poker_env.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace poker_ppo {

int cmd_train(IPokerEnvironmentFactory& factory,
              const PokerConfig&        poker_cfg,
              torch::Device             device,
              PPOTrainer::Strategy      strategy)
{
    PPOTrainer trainer(factory, device);
    trainer.set_rollout_strategy(strategy);

    const PPOConfig& ppo_cfg = config::kPPOConfig;
    const BetConfig& bet_cfg = config::kBetConfig;

    // Throw-away env to query obs/action dims for league/BR networks.
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
    // Matches reward_norm = 10 * big_blind in poker_env.
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

    // League/BR run synchronously — wall-clock cost shows up as gaps
    // between [update K] lines, not added to rollout/update timings.
    // Cadences are in env steps so they survive batch-shape changes;
    // interval-crossing (not modulo) since global_step rarely lands on
    // exact multiples.
    constexpr int64_t league_eval_every_steps = 2'457'600;
    int64_t last_league_eval_step = 0;
    int64_t last_br_eval_step     = 0;

    // Disable the periodic evals for a pure as-fast-as-possible training run.
    // BR especially is a big stall (it trains a full multi-update exploiter);
    // league is a frequent ~20s match. Neither affects the trained policy.
    const bool no_br     = std::getenv("POKER_PPO_NO_BR")     != nullptr;
    const bool no_league = std::getenv("POKER_PPO_NO_LEAGUE") != nullptr;

    const BestResponseConfig& br_cfg = config::kBRConfig;

    std::unique_ptr<BestResponseEvaluator> br_eval;
    if (br_cfg.enabled && !no_br) {
        br_eval = std::make_unique<BestResponseEvaluator>(
            factory, bet_cfg,
            obs_dim, action_count,
            ppo_cfg.hidden_dim, ppo_cfg.num_layers,
            ppo_cfg.hist, ppo_cfg.round_summary,
            br_cfg, device);
        std::cout << "Best-response evaluator: ON  (every "
                  << br_cfg.eval_every_steps << " env steps, "
                  << br_cfg.num_exploiter_seeds << " seeds × "
                  << br_cfg.updates_per_eval << " exploiter updates, "
                  << br_cfg.eval_hands << "-hand eval match)\n";
    } else {
        std::cout << "Best-response evaluator: OFF\n";
    }

    auto action_label = [&](int a) -> std::string {
        if (a == 0) return "F";
        if (a == 1) return "C";
        const int raise_idx = a - 2;
        const int n_pot     = static_cast<int>(poker_cfg.game.pot_fractions.size());
        if (raise_idx < n_pot) return "R" + std::to_string(raise_idx);
        return "AI";
    };

    // CSVs at runs/<timestamp>/, tailed by tools/plot_live.py.
    MetricsLogger metrics(make_run_dir());
    std::cout << "Metrics dir: " << metrics.run_dir() << "\n"
              << "  (live plots: `python tools/plot_live.py "
              << metrics.run_dir() << "`)\n";

    // Resume a cut-off run: POKER_PPO_RESUME=<ckpt_dir> loads network +
    // optimizer + counters before training (continues from the saved
    // update; opponent pool refills from empty). Periodic checkpoints are
    // written to this run's dir; cadence from config or
    // POKER_PPO_CHECKPOINT_STEPS.
    if (const char* r = std::getenv("POKER_PPO_RESUME")) {
        trainer.load_checkpoint(r);
    }
    int64_t ckpt_steps = ppo_cfg.checkpoint_every_steps;
    if (const char* e = std::getenv("POKER_PPO_CHECKPOINT_STEPS")) {
        ckpt_steps = std::atoll(e);
    }
    trainer.set_checkpoint(metrics.run_dir() + "/ckpt", ckpt_steps);
    if (ckpt_steps > 0) {
        std::cout << "Checkpointing: every " << ckpt_steps << " steps → "
                  << metrics.run_dir() << "/ckpt"
                  << "  (resume: POKER_PPO_RESUME=<dir>)\n";
    }

    trainer.set_log_callback([&](const PPOTrainer::UpdateStats& s) {
        metrics.log_update(s);

        if (s.update % 10 == 0) {
            std::cout << std::fixed << std::setprecision(4)
                      << "update=" << s.update
                      << "  step="   << s.global_step
                      << "  pg="     << s.policy_loss
                      << "  vf="     << s.value_loss
                      << "  H="      << s.entropy
                      << "  kl="     << s.approx_kl
                      << "  clip="   << s.clip_fraction
                      << "  ev="     << s.explained_variance
                      << "  lr="     << std::setprecision(6) << s.learning_rate
                      << "\n"
                      << std::defaultfloat << std::setprecision(6);
        }

        if (!no_league && s.update > 0 &&
            s.global_step - last_league_eval_step >= league_eval_every_steps) {
            last_league_eval_step = s.global_step;
            using clock = std::chrono::steady_clock;
            using ms    = std::chrono::duration<double, std::milli>;

            auto t_eval0 = clock::now();
            auto results = league.evaluate(trainer.network());
            const double eval_ms = ms(clock::now() - t_eval0).count();

            metrics.log_league(s.update, s.global_step, results);

            std::cout << "\n[league eval @ update " << s.update
                      << "  duration=" << eval_ms << "ms]\n";
            league.print_results(results);

            // pair_all_in folds ~94% preflop, so a healthy policy shows
            // high raise/all-in and low fold against it.
            auto print_action_mix = [&](const char* anchor_name) {
                int idx = -1;
                for (size_t i = 0; i < results.size(); ++i) {
                    if (results[i].anchor_name == anchor_name) {
                        idx = static_cast<int>(i);
                        break;
                    }
                }
                if (idx < 0) return;

                const auto& r = results[idx];
                int64_t total = 0;
                for (auto c : r.action_counts_a) total += c;
                if (total <= 0) return;

                std::cout << "  action mix (vs " << anchor_name << "):"
                          << std::fixed << std::setprecision(1);
                for (size_t a = 0; a < r.action_counts_a.size(); ++a) {
                    const float pct = 100.0f
                        * static_cast<float>(r.action_counts_a[a])
                        / static_cast<float>(total);
                    std::cout << "  " << action_label(static_cast<int>(a))
                              << "=" << pct << "%";
                }
                std::cout.unsetf(std::ios::fixed);
                std::cout << "\n";
            };
            if (!results.empty()) {
                print_action_mix("uniform");
                print_action_mix("pair_all_in");
            }
            std::cout << "\n";
        }

        if (br_eval && s.update > 0 &&
            s.global_step - last_br_eval_step >= br_cfg.eval_every_steps) {
            last_br_eval_step = s.global_step;
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

    if (!no_league) {
        std::cout << "\nFinal league evaluation...\n";
        auto final_results = league.evaluate(trainer.network());
        metrics.log_league(/*update=*/-1, /*step=*/-1, final_results);
        league.print_results(final_results);
    }

    // Canonical copy lives in the run dir (immune to later runs — a fixed
    // filename once cost us a 600M-step model overwritten by a 37M-step
    // experiment arm); the legacy fixed name stays as a "latest" pointer
    // for play tooling.
    const std::string run_model_path = metrics.run_dir() + "/model.pt";
    trainer.save(run_model_path);
    const std::string model_path =
        std::string("poker_ppo_model_") + std::string(poker_cfg.game.name) + ".pt";
    trainer.save(model_path);
    std::cout << "\nModel saved to " << run_model_path
              << " (and latest-pointer " << model_path << ")\n";
    return 0;
}

int cmd_play(IPokerEnvironmentFactory& factory,
             const BetConfig&          bet_cfg,
             torch::Device             device,
             const std::string&        model_path)
{
    if (model_path.empty()) {
        std::cerr << "--play requires a model path: ./poker_ppo --play <path>\n";
        return 1;
    }
    std::cerr << "[play] loading " << model_path << "\n";

    PPOTrainer trainer(factory, device);
    trainer.load(model_path);
    auto& net = trainer.network();
    net->eval();

    // Need PokerEnvironment, not just IPokerEnvironment, for the state
    // accessors (hole_cards, pot, ...).
    auto env_base = factory.create(bet_cfg);
    auto* env = dynamic_cast<PokerEnvironment*>(env_base.get());
    if (!env) {
        std::cerr << "ERR factory did not produce PokerEnvironment\n";
        return 1;
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
        const auto mask  = env->legal_action_mask();
        const auto m_acc = mask.accessor<float, 1>();
        std::cout << "mask " << mask.size(0);
        for (int i = 0; i < mask.size(0); ++i)
            std::cout << " " << (m_acc[i] > 0.5f ? 1 : 0);
        std::cout << "\n";
        const auto obs   = env->observation();
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

    // Deployment-time sharpening (DeepNash-style): the training policy is
    // an entropy-regularised QRE whose raise mass is thinly spread across
    // the 12 size slots (audited: trash hands carry ~67% total raise mass
    // at ~5% per size — noise, not beliefs). Dropping low-probability
    // actions and renormalising removes that noise while leaving
    // concentrated value-raises intact. Defaults: min_p=0.10, temp=1.0;
    // POKER_PPO_PLAY_MIN_P=0 restores the raw training policy.
    const float play_min_p = [] {
        const char* e = std::getenv("POKER_PPO_PLAY_MIN_P");
        return e ? static_cast<float>(std::atof(e)) : 0.10f;
    }();
    const float play_temp = [] {
        const char* e = std::getenv("POKER_PPO_PLAY_TEMP");
        const float t = e ? static_cast<float>(std::atof(e)) : 1.0f;
        return t > 0.0f ? t : 1.0f;
    }();
    std::cerr << "[play] sampling: min_p=" << play_min_p
              << " temp=" << play_temp
              << " (POKER_PPO_PLAY_MIN_P / POKER_PPO_PLAY_TEMP)\n";

    auto emit_model = [&]() {
        torch::NoGradGuard ng;
        auto obs  = env->observation().unsqueeze(0).to(device);
        auto mask = env->legal_action_mask().unsqueeze(0).to(device);
        auto [logits, critic_raw] = net->forward(obs);
        (void)critic_raw;  // VRPO: Q(s,·); report the masked expected value V̄
        const auto masked = logits + (1.0f - mask) * kIllegalActionLogit;
        auto probs = torch::softmax(masked / play_temp, -1)
                         .squeeze(0).to(torch::kCPU).contiguous();
        // Argmax for diagnostics (sharpening never changes it).
        const auto greedy = std::get<1>(probs.max(0)).item<int64_t>();
        if (play_min_p > 0.0f) {
            auto kept = probs * (probs >= play_min_p).to(torch::kFloat32);
            const float z = kept.sum().item<float>();
            if (z > 0.0f) {
                probs = kept / z;
            } else {
                probs = torch::zeros_like(probs);  // all filtered → argmax
                probs[greedy] = 1.0f;
            }
        }
        // Stochastic sample from the (possibly sharpened) distribution.
        const auto sampled =
            probs.multinomial(1).item<int64_t>();
        const float value  = net->get_state_value(obs, mask).item<float>();
        std::cout << "sampled "  << sampled << "\n";
        std::cout << "greedy "   << greedy  << "\n";
        std::cout << "value "    << value << "\n";
        const auto p_acc = probs.accessor<float, 1>();
        std::cout << "probs " << probs.size(0);
        for (int i = 0; i < probs.size(0); ++i)
            std::cout << " " << p_acc[i];
        std::cout << "\n";
        std::cout << "OK" << std::endl;
    };

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
    return 0;
}

int cmd_benchmark(IPokerEnvironmentFactory& factory,
                  torch::Device             device,
                  int                       iters)
{
    std::cout << "\n[benchmark mode] comparing serial / threadpool\n";
    PPOTrainer trainer(factory, device);
    trainer.benchmark_rollouts(iters, /*warmup=*/3);
    return 0;
}

int cmd_br_eval(IPokerEnvironmentFactory& factory,
                const PokerConfig&        poker_cfg,
                torch::Device             device,
                const std::string&        model_path)
{
    if (model_path.empty()) {
        std::cerr << "--br-eval requires a model path\n";
        return 1;
    }

    const PPOConfig& ppo_cfg = config::kPPOConfig;
    const BetConfig& bet_cfg = config::kBetConfig;

    // Load the target net (architecture from current config — must match
    // how the model was trained, incl. POKER_PPO_HISTORY_ENCODER).
    PPOTrainer trainer(factory, device);
    std::cerr << "[br-eval] loading " << model_path << "\n";
    trainer.load(model_path);
    trainer.network()->eval();

    int obs_dim, action_count;
    {
        auto tmp     = factory.create(bet_cfg);
        obs_dim      = tmp->obs_dim();
        action_count = tmp->bet_config().action_count();
    }

    BestResponseConfig br_cfg = config::kBRConfig;
    br_cfg.enabled = true;  // creation is unconditional here
    BestResponseEvaluator br(factory, bet_cfg, obs_dim, action_count,
                             ppo_cfg.hidden_dim, ppo_cfg.num_layers,
                             ppo_cfg.hist, ppo_cfg.round_summary,
                             br_cfg, device);

    // Read the EFFECTIVE config from the evaluator, not the static br_cfg —
    // the POKER_PPO_BR_* overrides are applied inside the ctor and only
    // land in the evaluator's own copy, so printing br_cfg here reported
    // pre-override defaults regardless of the env vars actually in force.
    const BestResponseConfig& eff = br.config();
    std::cout << "[br-eval] training exploiter: "
              << eff.num_exploiter_seeds << " seed(s) × "
              << eff.updates_per_eval << " updates, "
              << eff.eval_hands << "-hand match"
              << "  (overrides: POKER_PPO_BR_SEEDS/_UPDATES/_LR/_ENT)\n";

    auto r = br.evaluate(trainer.network(), /*update=*/0, /*global_step=*/0);

    std::cout << std::fixed << std::setprecision(3)
              << "\n══════════ best-response (exploitability) ══════════\n"
              << "  bb/hand  max=" << r.bb_per_hand_a
              << "  mean="  << r.bb_per_hand_mean
              << "  min="   << r.bb_per_hand_min
              << "  std="   << r.bb_per_hand_std << "\n"
              << "  seeds=" << r.num_seeds
              << "  best-win%=" << std::setprecision(1) << (100.0f * r.win_rate_a)
              << "  hands="  << r.num_hands
              << "  duration=" << std::setprecision(0) << r.wall_ms << "ms\n"
              << "  lower bound on exploitability = max bb/hand "
              << "(higher = more exploitable)\n"
              << std::defaultfloat << std::setprecision(6);
    return 0;
}

int cmd_lbr_eval(IPokerEnvironmentFactory& factory,
                 const PokerConfig&        poker_cfg,
                 torch::Device             device,
                 const std::string&        model_path)
{
    if (model_path.empty()) {
        std::cerr << "--lbr-eval requires a model path\n";
        return 1;
    }
    const BetConfig& bet_cfg = config::kBetConfig;

    PPOTrainer trainer(factory, device);
    std::cerr << "[lbr-eval] loading " << model_path << "\n";
    trainer.load(model_path);
    trainer.network()->eval();

    LBRConfig cfg;
    cfg.seed = 0x1B20BEEFull;  // fixed for reproducibility
    if (const char* e = std::getenv("POKER_PPO_LBR_HANDS")) {
        const int v = std::atoi(e);
        if (v > 0) cfg.num_hands = v;
    }
    if (const char* e = std::getenv("POKER_PPO_LBR_MC")) {
        const int v = std::atoi(e);
        if (v > 0) cfg.equity_mc_samples = v;
    }
    if (std::getenv("POKER_PPO_LBR_NO_RAISE") != nullptr) {
        cfg.enable_raises = false;  // v1 {fold,call} only
    }

    std::cout << "[lbr-eval] Local Best Response ("
              << (cfg.enable_raises ? "v2: {fold,call,river-raise}"
                                    : "v1: {fold,call}")
              << "), " << cfg.num_hands << " hands, equity_mc="
              << cfg.equity_mc_samples << "\n";

    LBREvaluator lbr(factory, bet_cfg, cfg, device);
    auto r = lbr.evaluate(trainer.network());

    std::cout << std::fixed << std::setprecision(3)
              << "\n══════════ LBR (exploitability lower bound) ══════════\n"
              << "  bb/hand  = " << r.bb_per_hand
              << "   (mbb/hand=" << std::setprecision(1) << r.mbb_per_hand << ")\n"
              << std::setprecision(3)
              << "  lbr-win%=" << std::setprecision(1) << (100.0 * r.lbr_win_rate)
              << "  hands="    << r.num_hands
              << "  duration=" << std::setprecision(0) << r.wall_ms << "ms\n"
              << "  exploitability ≥ bb/hand (≥0 for a true BR; v1 {f,c} is a\n"
              << "  conservative attacker — a positive value is a hard exploit)\n"
              << std::defaultfloat << std::setprecision(6);
    return 0;
}

} // namespace poker_ppo
