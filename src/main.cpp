//   ./poker_ppo                                  train (default)
//   ./poker_ppo --benchmark [iters]              rollout A/B bench
//   ./poker_ppo --play <model_path>              interactive REPL
//   ./poker_ppo --strategy {serial|threadpool}   rollout strategy

#include "commands.h"
#include "poker_env.h"

#include <iostream>
#include <string>
#include <string_view>

using namespace poker_ppo;

namespace {

void print_game(const ::Game::DefaultGameConfig& g) {
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

struct CliOptions {
    bool        benchmark_mode  = false;
    bool        play_mode       = false;
    std::string play_model_path;
    int         benchmark_iters = 20;
    std::string strategy        = "threadpool";
};

bool parse_cli(int argc, char** argv, CliOptions& out) {
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--benchmark") {
            out.benchmark_mode = true;
            if (i + 1 < argc) {
                try { out.benchmark_iters = std::stoi(argv[i + 1]); ++i; }
                catch (...) { /* leave default */ }
            }
        } else if (a == "--play") {
            out.play_mode = true;
            if (i + 1 < argc) { out.play_model_path = argv[i + 1]; ++i; }
        } else if (a == "--strategy") {
            if (i + 1 < argc) { out.strategy = argv[i + 1]; ++i; }
        } else {
            std::cerr << "unknown argument '" << a << "'\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);  // unbuffered for PTY/log capture

    CliOptions opt;
    if (!parse_cli(argc, argv, opt)) return 1;

    PokerConfig poker_cfg = kPokerConfig;
    poker_cfg.game.validate();

    print_game(poker_cfg.game);
    {
        const PPOConfig& p = config::kPPOConfig;
        std::cout << "Bet-history attention encoder: "
                  << (p.hist.enabled ? "ON" : "OFF") << "\n"
                  << "Round-summary block          : "
                  << (p.round_summary.enabled ? "ON" : "OFF") << "\n"

        << "Opponent pool                : "
                  << (p.opp_pool.enabled ? "ON" : "OFF");
        if (p.opp_pool.enabled) {
            std::cout << "  (size=" << p.opp_pool.max_size
                      << ", snapshot_every=" << p.opp_pool.snapshot_every
                      << ", warmup=" << p.opp_pool.warmup_updates
                      << ", p_use_pool=" << p.opp_pool.p_use_pool
                      << ", max_unique=" << p.opp_pool.max_unique_per_rollout << ")";
        }
        std::cout << "\n";
    }

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::mps::is_available() ? torch::kMPS : torch::kCPU;
    //torch::Device device = torch::kCPU;
    std::cout << "Using device: " << device << "\n";

    PokerEnvironmentFactory factory(poker_cfg);

    if (opt.play_mode) {
        return cmd_play(factory, config::kBetConfig, device, opt.play_model_path);
    }
    if (opt.benchmark_mode) {
        return cmd_benchmark(factory, device, opt.benchmark_iters);
    }

    PPOTrainer::Strategy strategy;
    if (opt.strategy == "serial") {
        strategy = PPOTrainer::Strategy::Serial;
    } else if (opt.strategy == "threadpool") {
        strategy = PPOTrainer::Strategy::Threadpool;
    } else {
        std::cerr << "unknown --strategy '" << opt.strategy
                  << "' (expected serial|threadpool)\n";
        return 1;
    }
    std::cout << "Rollout strategy: " << opt.strategy << "\n";
    return cmd_train(factory, poker_cfg, device, strategy);
}
