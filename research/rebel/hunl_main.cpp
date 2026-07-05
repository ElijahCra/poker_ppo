// HUNL ReBeL stage-2 validation ladder:
//   rebel_hunl kernels                       fast-vs-brute kernel equivalence
//   rebel_hunl river [T] [--sparse]          river subgame: internal BR → 0
//   rebel_hunl turn  [T_outer] [T_inner]     turn decomposition, exact river
//                                            leaves (sparse actions), BR → 0
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "config.h"
#include "hunl_kernels.h"
#include "hunl_solver.h"
#include "hunl_value.h"
#include "poker_env.h"

using namespace rebel_hunl;
using poker_ppo::PokerEnvironment;

namespace {

// deal-agnostic river/turn root: reset, then check/call to the target street
void advance_to_round(PokerEnvironment& env, int target) {
    env.reset();
    while (!env.is_terminal() && env.round() < target) env.step(1);
}

std::vector<double> random_range(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<double> r(kCombos);
    for (double& x : r) {
        const double v = u(rng);
        x = v * v;   // skewed weights exercise the card-removal terms harder
    }
    return r;
}

int run_kernels() {
    std::mt19937 rng(7);
    double worst = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        PokerEnvironment env(poker_ppo::kPokerConfig,
                             poker_ppo::config::kBetConfig, 1000 + trial);
        advance_to_round(env, 3);
        uint8_t board[5];
        for (int i = 0; i < 5; ++i)
            board[i] = static_cast<uint8_t>(env.community_card(i));
        auto opp = random_range(rng);
        std::vector<uint8_t> valid;
        board_valid(board, 5, valid);

        std::vector<double> a, b;
        compat_mass(opp, valid, a);
        compat_mass_brute(opp, valid, b);
        for (int i = 0; i < kCombos; ++i)
            worst = std::max(worst, std::fabs(a[i] - b[i]));

        showdown_cfv(opp, board, 500.0, a);
        showdown_cfv_brute(opp, board, 500.0, b);
        for (int i = 0; i < kCombos; ++i)
            worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    std::printf("kernels: max |fast - brute| over 20 boards = %.3e  %s\n",
                worst, worst < 1e-9 ? "PASS" : "FAIL");
    return worst < 1e-9 ? 0 : 1;
}

void report(HunlSolver& s, int t, int big_blind) {
    const double e = s.exploitability();
    std::printf("  T=%-6d subgame expl = %10.2f chips/hand  (%.4f bb)\n", t,
                e, e / big_blind);
    std::fflush(stdout);
}

int run_river(int T, bool sparse) {
    PokerEnvironment env(poker_ppo::kPokerConfig,
                         poker_ppo::config::kBetConfig, 42);
    advance_to_round(env, 3);
    std::printf("river subgame: board");
    for (int i = 0; i < 5; ++i) std::printf(" %d", env.community_card(i));
    std::printf("  pot=%d  actions=%s\n", env.pot(),
                sparse ? "sparse{f,c,pot,allin}" : "full");
    std::mt19937 rng(3);
    HunlPBS beta;
    beta.r0 = random_range(rng);
    beta.r1 = random_range(rng);
    std::vector<int> allowed;
    if (sparse) allowed = {0, 1, 7, 13};
    HunlSolver s(env, beta, nullptr, allowed);
    std::printf("  tree: %zu nodes\n", s.nodes().size());
    const int bb = env.game_config().big_blind;
    for (int t = 1; t <= T; ++t) {
        s.iterate(t);
        if (t == 20 || t == 100 || t == T) report(s, t, bb);
    }
    return 0;
}

int run_turn(int T_outer, int T_inner) {
    PokerEnvironment env(poker_ppo::kPokerConfig,
                         poker_ppo::config::kBetConfig, 42);
    advance_to_round(env, 2);
    std::printf("turn subgame: board");
    for (int i = 0; i < 4; ++i) std::printf(" %d", env.community_card(i));
    std::printf("  pot=%d  actions=sparse{f,c,pot,allin}  inner=%d\n",
                env.pot(), T_inner);
    std::mt19937 rng(5);
    HunlPBS beta;
    beta.r0 = random_range(rng);
    beta.r1 = random_range(rng);
    const std::vector<int> allowed = {0, 1, 7, 13};
    ExactStreetOracle oracle(T_inner, allowed);
    HunlSolver s(env, beta, &oracle, allowed);
    s.refresh_every = 5;   // exact-oracle refreshes are the dominant cost
    std::printf("  tree: %zu nodes\n", s.nodes().size());
    const int bb = env.game_config().big_blind;
    for (int t = 1; t <= T_outer; ++t) {
        s.iterate(t);
        if (t == 10 || t == 25 || t == 50 || t == T_outer)
            report(s, t, bb);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "kernels";
    if (mode == "kernels") return run_kernels();
    if (mode == "river") {
        int T = argc > 2 ? std::atoi(argv[2]) : 400;
        const bool sparse =
            argc > 3 && std::strcmp(argv[3], "--sparse") == 0;
        return run_river(T, sparse);
    }
    if (mode == "turn") {
        const int To = argc > 2 ? std::atoi(argv[2]) : 100;
        const int Ti = argc > 3 ? std::atoi(argv[3]) : 100;
        return run_turn(To, Ti);
    }
    if (mode == "train_turn") {
        EndgameConfig cfg;
        if (argc > 2) cfg.epochs = std::atoi(argv[2]);
        if (argc > 3) cfg.episodes = std::atoi(argv[3]);
        EndgameTrainer tr(cfg);
        tr.run();
        return 0;
    }
    std::fprintf(stderr, "mode must be kernels|river|turn|train_turn\n");
    return 1;
}
