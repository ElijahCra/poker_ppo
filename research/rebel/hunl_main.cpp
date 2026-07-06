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
#include "hunl_gpu.h"
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

// Encoding cross-check: the solver's terminal payoff model (contribs +
// kernels' winner logic) vs env.terminal_utility for the env's actually
// dealt hands, across many boards/paths. Closes the "internal BR shares the
// tree encoding" loophole with ground truth from outside the solver.
int run_check_terminals(int n_states, int start = 0) {
    const std::vector<int> allowed = {0, 1, 7, 13};
    double worst_fold = 0.0, worst_show = 0.0, worst_allin = 0.0;
    long n_fold = 0, n_show = 0, n_allin = 0;
    for (int s = start; s < start + n_states; ++s) {
        if (std::getenv("REBEL_DEBUG"))
            std::fprintf(stderr, "  [dbg] state %d\n", s);
        PokerEnvironment env(poker_ppo::kPokerConfig,
                             poker_ppo::config::kBetConfig, 5000 + s);
        advance_to_round(env, s % 2 == 0 ? 3 : 2);
        if (env.is_terminal()) continue;
        std::mt19937 rng(s);
        HunlPBS beta;
        beta.r0 = random_range(rng);
        beta.r1 = random_range(rng);
        HunlSolver sol(env, beta, nullptr, allowed);
        const auto h0 = env.hole_cards(0);
        const auto h1 = env.hole_cards(1);
        const auto& ct = ComboTable::get();
        const int c0 = ct.id[h0[0]][h0[1]];
        const int c1 = ct.id[h1[0]][h1[1]];
        for (size_t i = 0; i < sol.nodes().size(); ++i) {
            const auto& nd = sol.nodes()[i];
            if (nd.kind == HunlSolver::Node::Decision ||
                nd.kind == HunlSolver::Node::StreetEnd)
                continue;
            if (std::getenv("REBEL_DEBUG")) {
                std::fprintf(stderr, "  [dbg] node %zu kind %d path:", i,
                             (int)nd.kind);
                for (int a : nd.path) std::fprintf(stderr, " %d", a);
                std::fprintf(stderr, "\n");
            }
            env.push_state();
            double last_reward = 0.0;
            for (int a : nd.path) last_reward = env.step(a).reward;
            // terminal_utility returns the REALIZED getUtility — for all-in
            // terminals that's a bogus incomplete-board showdown; the equity
            // value only flows through the step reward (P0 frame, scaled by
            // reward_norm = 10*bb).
            const double truth =
                nd.kind == HunlSolver::Node::AllinShowdown
                    ? last_reward * 10.0 * env.game_config().big_blind
                    : env.terminal_utility(0);
            double model = 0.0;
            if (nd.kind == HunlSolver::Node::Fold) {
                model = nd.player == 1
                    ? static_cast<double>(nd.contrib[1])
                    : -static_cast<double>(nd.contrib[0]);
                worst_fold = std::max(worst_fold, std::fabs(model - truth));
                ++n_fold;
            } else if (nd.kind == HunlSolver::Node::Showdown) {
                uint8_t b5[5];
                for (int b = 0; b < 5; ++b)
                    b5[b] = static_cast<uint8_t>(env.community_card(b));
                const int r0 = combo_rank(c0, b5), r1 = combo_rank(c1, b5);
                model = r0 > r1 ? nd.contrib[1]
                       : r0 < r1 ? -static_cast<double>(nd.contrib[0]) : 0.0;
                worst_show = std::max(worst_show, std::fabs(model - truth));
                ++n_show;
            } else {   // AllinShowdown at the turn: enumerate rivers
                const auto& bd = sol.board();
                double ev = 0.0;
                int cnt = 0;
                for (int c = 0; c < kCards; ++c) {
                    bool dead = false;
                    for (int b = 0; b < sol.board_count(); ++b)
                        if (bd[b] == c) dead = true;
                    if (c == h0[0] || c == h0[1] || c == h1[0] || c == h1[1])
                        dead = true;
                    if (dead) continue;
                    std::array<uint8_t, 5> b5{};
                    for (int b = 0; b < sol.board_count(); ++b) b5[b] = bd[b];
                    b5[sol.board_count()] = static_cast<uint8_t>(c);
                    const int r0 = combo_rank(c0, b5.data());
                    const int r1 = combo_rank(c1, b5.data());
                    ev += r0 > r1 ? nd.contrib[1]
                        : r0 < r1 ? -static_cast<double>(nd.contrib[0]) : 0.0;
                    ++cnt;
                }
                model = cnt > 0 ? ev / cnt : 0.0;
                worst_allin = std::max(worst_allin, std::fabs(model - truth));
                ++n_allin;
            }
            env.pop_state();
        }
    }
    std::printf("terminal payoff model vs env.terminal_utility:\n"
                "  fold      n=%-6ld max|diff|=%.3f chips\n"
                "  showdown  n=%-6ld max|diff|=%.3f chips\n"
                "  allin     n=%-6ld max|diff|=%.3f chips (int rounding + "
                "equity method)\n",
                n_fold, worst_fold, n_show, worst_show, n_allin, worst_allin);
    const bool pass = worst_fold < 0.5 && worst_show < 0.5 && worst_allin < 2.0;
    std::printf("  %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Batched-vs-scalar solver equivalence: B random river situations solved by
// BatchRiverSolver (fp32 tensors, GPU on the box / CPU torch here) and by
// the fp64 CPU HunlSolver, comparing per-combo root values in pot units.
int run_gpu_check(int B, int T) {
    const std::vector<int> allowed = {0, 1, 7, 13};
    const torch::Device dev =
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::mt19937 rng(11);
    PokerEnvironment env(poker_ppo::kPokerConfig,
                         poker_ppo::config::kBetConfig, 77);

    std::vector<RiverSpec> specs;
    std::vector<std::vector<std::array<std::vector<double>, 2>>> cpu_v;
    std::vector<std::vector<std::array<std::vector<double>, 2>>> cpu_m;
    TreeShape shape;
    bool have_shape = false;
    double pot0 = 0.0;
    int tries = 0;
    while (static_cast<int>(specs.size()) < B && ++tries < 20 * B) {
        advance_to_round(env, 3);
        if (env.is_terminal()) continue;
        RiverSpec sp;
        for (int i = 0; i < 5; ++i)
            sp.board[i] = static_cast<uint8_t>(env.community_card(i));
        sp.r0 = random_range(rng);
        sp.r1 = random_range(rng);
        HunlPBS beta;
        beta.r0 = sp.r0;
        beta.r1 = sp.r1;
        HunlSolver cpu(env, beta, nullptr, allowed);
        auto sh = TreeShape::from(cpu);
        if (!have_shape) {
            shape = sh;
            have_shape = true;
            pot0 = 2.0 * cpu.nodes()[0].contrib[0];
        } else if (sh.signature != shape.signature) {
            continue;   // rare divergent topology → grouped out
        }
        for (const auto& nd : cpu.nodes()) {
            sp.node_contrib0.push_back(nd.contrib[0]);
            sp.node_contrib1.push_back(nd.contrib[1]);
        }
        // CPU reference solve
        for (int t = 1; t <= T; ++t) cpu.iterate(t);
        std::vector<std::array<std::vector<double>, 2>> v1(1), m1(1);
        cpu.root_values(v1[0], m1[0]);
        cpu_v.push_back(std::move(v1));
        cpu_m.push_back(std::move(m1));
        specs.push_back(std::move(sp));
    }
    (void)pot0;

    const bool f64 = std::getenv("REBEL_GPU_F64") != nullptr;
    BatchRiverSolver gpu(shape, specs, dev,
                         f64 ? torch::kDouble : torch::kFloat);
    for (int t = 1; t <= T; ++t) gpu.iterate(t);
    std::vector<std::array<std::vector<double>, 2>> gv, gm;
    gpu.root_values(gv, gm);

    double worst = 0.0, mean = 0.0;
    long cnt = 0;
    size_t wb = 0;
    int wp = 0, wi = 0;
    for (size_t b = 0; b < specs.size(); ++b) {
        const double pot = 2.0 * specs[b].node_contrib0[0];
        for (int p = 0; p < 2; ++p)
            for (int i = 0; i < kCombos; ++i) {
                if (cpu_m[b][0][p][i] < 0.5 || gm[b][p][i] < 0.5) continue;
                const double d =
                    std::fabs(cpu_v[b][0][p][i] - gv[b][p][i]) / pot;
                if (d > worst) {
                    worst = d;
                    wb = b;
                    wp = p;
                    wi = i;
                }
                mean += d;
                ++cnt;
            }
    }
    mean = cnt > 0 ? mean / cnt : 0.0;
    // fp64 = exact-equivalence bar; fp32 = accumulation-drift bar (validated
    // exact under fp64; ~0.5% of pot drift over hundreds of iterations is
    // far below the value net's error scale)
    const double bar = f64 ? 1e-3 : 2e-2;
    {
        // argmax context: opponent compatible mass at the offending combo
        std::vector<uint8_t> vb;
        board_valid(specs[wb].board.data(), 5, vb);
        const auto& opp = wp == 0 ? specs[wb].r1 : specs[wb].r0;
        double s = 0.0;
        for (int i = 0; i < kCombos; ++i)
            if (vb[i]) s += opp[i];
        std::vector<double> nrm(kCombos, 0.0);
        for (int i = 0; i < kCombos; ++i)
            if (vb[i] && s > 0) nrm[i] = opp[i] / s;
        std::vector<double> mass;
        compat_mass(nrm, vb, mass);
        std::printf("  worst: spec=%zu p=%d combo=%d  cpu=%.5f gpu=%.5f "
                    "(pot units)  opp-mass=%.3e\n",
                    wb, wp, wi,
                    cpu_v[wb][0][wp][wi] / (2.0 * specs[wb].node_contrib0[0]),
                    gv[wb][wp][wi] / (2.0 * specs[wb].node_contrib0[0]),
                    mass[wi]);
    }
    std::printf("gpu_check: B=%zu T=%d device=%s dtype=%s  |Δv|/pot "
                "mean=%.3e max=%.3e  %s\n",
                specs.size(), T, dev.is_cuda() ? "cuda" : "cpu",
                f64 ? "f64" : "f32", mean, worst,
                worst < bar ? "PASS" : "FAIL");
    return worst < bar ? 0 : 1;
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

// Validate the validator: exact-oracle turn solve, judged by BOTH the
// leaf-valued internal BR and the TRUE composed two-street BR. Composed
// should be ≥ internal (the BR gains river freedom) and both should shrink
// with T — the HUNL analogue of the Leduc exact ladder.
int run_turn_true(int T_outer, int T_river) {
    PokerEnvironment env(poker_ppo::kPokerConfig,
                         poker_ppo::config::kBetConfig, 42);
    advance_to_round(env, 2);
    std::mt19937 rng(5);
    HunlPBS beta;
    beta.r0 = random_range(rng);
    beta.r1 = random_range(rng);
    const std::vector<int> allowed = {0, 1, 7, 13};
    ExactStreetOracle oracle(T_river / 2, allowed);
    const int bb = env.game_config().big_blind;
    for (int T : {T_outer / 2, T_outer}) {
        HunlSolver s(env, beta, &oracle, allowed);
        s.refresh_every = 5;
        for (int t = 1; t <= T; ++t) s.iterate(t);
        const double internal = s.exploitability();
        const double composed = s.exploitability_composed(T_river);
        std::printf("  T=%-4d internal(leaf-valued) = %.4f bb   "
                    "TRUE composed = %.4f bb\n",
                    T, internal / bb, composed / bb);
        std::fflush(stdout);
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
    if (mode == "gpu_check")
        return run_gpu_check(argc > 2 ? std::atoi(argv[2]) : 16,
                             argc > 3 ? std::atoi(argv[3]) : 200);
    if (mode == "check_terminals")
        return run_check_terminals(argc > 2 ? std::atoi(argv[2]) : 20,
                                   argc > 3 ? std::atoi(argv[3]) : 0);
    if (mode == "turn_true") {
        const int To = argc > 2 ? std::atoi(argv[2]) : 25;
        const int Tr = argc > 3 ? std::atoi(argv[3]) : 150;
        return run_turn_true(To, Tr);
    }
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
    if (mode == "train_turn" || mode == "train_river") {
        EndgameConfig cfg;
        cfg.river_only = (mode == "train_river");
        if (cfg.river_only) {
            cfg.episodes = 2000;   // direct river targets per epoch
            cfg.sgd_steps = 500;
        }
        if (argc > 2) cfg.epochs = std::atoi(argv[2]);
        if (argc > 3) cfg.episodes = std::atoi(argv[3]);
        if (const char* t = std::getenv("REBEL_THREADS"))
            cfg.threads = std::atoi(t);
        if (const char* g = std::getenv("REBEL_GPU_BATCH"))
            cfg.gpu_batch = std::atoi(g);
        if (const char* h = std::getenv("REBEL_HARVEST"))
            cfg.harvest = std::atoi(h);
        EndgameTrainer tr(cfg);
        tr.run();
        return 0;
    }
    std::fprintf(stderr, "mode must be kernels|river|turn|train_turn\n");
    return 1;
}
