#include "commands.h"

#include "best_response.h"
#include "league.h"
#include "metrics_logger.h"
#include "poker_env.h"

#include <chrono>
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
    constexpr int snapshot_every = 200;

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

    trainer.set_log_callback([&](const PPOTrainer::UpdateStats& s) {
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

    std::cout << "\nFinal league evaluation...\n";
    auto final_results = league.evaluate(trainer.network());
    metrics.log_league(/*update=*/-1, /*step=*/-1, final_results);
    league.print_results(final_results);

    const std::string model_path =
        std::string("poker_ppo_model_") + std::string(poker_cfg.game.name) + ".pt";
    trainer.save(model_path);
    std::cout << "\nModel saved to " << model_path << "\n";
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

    auto emit_model = [&]() {
        torch::NoGradGuard ng;
        auto obs  = env->observation().unsqueeze(0).to(device);
        auto mask = env->legal_action_mask().unsqueeze(0).to(device);
        auto [logits, value] = net->forward(obs);
        const auto masked = logits + (1.0f - mask) * kIllegalActionLogit;
        const auto probs  = torch::softmax(masked, -1).squeeze(0).to(torch::kCPU).contiguous();
        // Stochastic sample (matches the training rollout).
        const auto ar      = net->get_action(obs, mask);
        const auto sampled = ar.action.to(torch::kCPU).item<int64_t>();
        // Argmax for diagnostics.
        const auto greedy  = std::get<1>(probs.max(0)).item<int64_t>();
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

} // namespace poker_ppo
