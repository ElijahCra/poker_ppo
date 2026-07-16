// HUNL ReBeL stage-2 validation ladder:
//   rebel_hunl kernels                       fast-vs-brute kernel equivalence
//   rebel_hunl river [T] [--sparse]          river subgame: internal BR → 0
//   rebel_hunl turn  [T_outer] [T_inner]     turn decomposition, exact river
//                                            leaves (sparse actions), BR → 0
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "config.h"
#include "hunl_gpu.h"
#include "hunl_gpu_turn.h"
#include "hunl_kernels.h"
#include "hunl_play.h"
#include "hunl_solver.h"
#include "hunl_value.h"
#include "lbr.h"
#include "network.h"
#include "poker_env.h"

#include <unordered_map>

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

    // solve() path (f64 → CUDA graph, bitwise vs eager; f32 → fused NVRTC
    // kernel, different summation order → RM⁺ equilibrium-selection
    // spread). Judged against the CPU fp64 solver with the SAME bars as
    // the eager path — that's the truth anchor; solve-vs-eager is
    // informational.
    BatchRiverSolver gpu_g(shape, specs, dev,
                           f64 ? torch::kDouble : torch::kFloat);
    gpu_g.solve(T);
    std::vector<std::array<std::vector<double>, 2>> ggv, ggm;
    gpu_g.root_values(ggv, ggm);
    double gworst = 0.0, sworst = 0.0, smean = 0.0;
    long scnt = 0;
    for (size_t b = 0; b < specs.size(); ++b) {
        const double pot = 2.0 * specs[b].node_contrib0[0];
        for (int p = 0; p < 2; ++p)
            for (int i = 0; i < kCombos; ++i) {
                if (cpu_m[b][0][p][i] > 0.5 && ggm[b][p][i] > 0.5) {
                    const double d =
                        std::fabs(cpu_v[b][0][p][i] - ggv[b][p][i]) / pot;
                    sworst = std::max(sworst, d);
                    smean += d;
                    ++scnt;
                }
                if (gm[b][p][i] < 0.5 || ggm[b][p][i] < 0.5) continue;
                gworst = std::max(gworst,
                                  std::fabs(gv[b][p][i] - ggv[b][p][i]) /
                                      pot);
            }
    }
    smean = scnt > 0 ? smean / scnt : 0.0;
    std::printf("  solve-vs-eager |Δv|/pot max=%.3e (informational)\n",
                gworst);
    std::printf("  solve-vs-cpu   |Δv|/pot mean=%.3e max=%.3e\n", smean,
                sworst);

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
    // Neither bar is bitwise: RM⁺'s clamp is discontinuous, so ~1e-16
    // rounding differences between implementations flip clamp decisions
    // and land a few low-info combos on different (equally valid)
    // equilibrium selections (measured 2026-07-09: f64 mean 9.7e-6, max
    // 5.9e-3 — IDENTICAL before/after the sparse-kernel + CUDA-graph
    // rewrite, i.e. pre-existing selection spread, not arithmetic error;
    // the historical 6.9e-5 was a different spec draw). The MEAN is the
    // arithmetic-honesty signal; the max rides equilibrium selection.
    const double bar = f64 ? 1e-2 : 2e-2;
    const double mean_bar = f64 ? 1e-4 : 1e-3;
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
    const bool pass = worst < bar && mean < mean_bar && sworst < bar &&
                      smean < mean_bar;
    std::printf("gpu_check: B=%zu T=%d device=%s dtype=%s  |Δv|/pot "
                "mean=%.3e max=%.3e  %s\n",
                specs.size(), T, dev.is_cuda() ? "cuda" : "cpu",
                f64 ? "f64" : "f32", mean, worst, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// Throughput: CPU worker pool vs the CUDA-graphed batch solver, same tree
// topology and T — the economics that decide where river solves run
// (train_river targets today; play-time re-solving once single-solve
// latency wins too).
int run_gpu_bench(int B, int T, int W) {
    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };
    const std::vector<int> allowed = {0, 1, 7, 13};
    const torch::Device dev =
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::mt19937 rng(11);
    PokerEnvironment env(poker_ppo::kPokerConfig,
                         poker_ppo::config::kBetConfig, 77);
    {   // HandRanks pre-warm (worker threads + spec building)
        const uint8_t warm[5] = {0, 5, 10, 15, 20};
        (void)combo_rank(ComboTable::get().id[25][30], warm);
    }

    std::vector<RiverSpec> specs;
    TreeShape shape;
    bool have_shape = false;
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
        } else if (sh.signature != shape.signature) {
            continue;
        }
        for (const auto& nd : cpu.nodes()) {
            sp.node_contrib0.push_back(nd.contrib[0]);
            sp.node_contrib1.push_back(nd.contrib[1]);
        }
        specs.push_back(std::move(sp));
    }

    // CPU pool: W workers, each solves fresh random river subgames of the
    // same topology until B are done (the train_river economics)
    torch::set_num_threads(1);
    auto t0 = clock::now();
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int w = 0; w < W; ++w)
        pool.emplace_back([&, w] {
            PokerEnvironment we(poker_ppo::kPokerConfig,
                                poker_ppo::config::kBetConfig, 500 + w);
            std::mt19937 wr(1000 + w);
            while (next.fetch_add(1) < B) {
                advance_to_round(we, 3);
                if (we.is_terminal()) continue;
                HunlPBS beta;
                beta.r0 = random_range(wr);
                beta.r1 = random_range(wr);
                HunlSolver s(we, beta, nullptr, allowed);
                for (int t = 1; t <= T; ++t) s.iterate(t);
            }
        });
    for (auto& th : pool) th.join();
    const double cpu_s = secs(t0, clock::now());

    // GPU: build (CPU-side prep) and graphed solve, timed separately
    auto t1 = clock::now();
    BatchRiverSolver gpu(shape, specs, dev, torch::kFloat);
    const double build_s = secs(t1, clock::now());
    auto t2 = clock::now();
    gpu.solve(T);
    const double solve_s = secs(t2, clock::now());
    std::vector<std::array<std::vector<double>, 2>> v, m;
    auto t3 = clock::now();
    gpu.root_values(v, m);
    const double read_s = secs(t3, clock::now());

    const double n = static_cast<double>(specs.size());
    std::printf("gpu_bench: B=%zu T=%d W=%d device=%s\n", specs.size(), T,
                W, dev.is_cuda() ? "cuda" : "cpu");
    std::printf("  cpu pool : %6.2fs  %7.1f targets/s\n", cpu_s, n / cpu_s);
    std::printf("  gpu build: %6.2fs  solve: %6.2fs  read: %5.2fs  "
                "%7.1f targets/s (solve-only %7.1f/s)\n",
                build_s, solve_s, read_s, n / (build_s + solve_s + read_s),
                n / solve_s);
    return 0;
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

// Gadget guarantee, judged from outside the gadget: after a gadgeted
// solve, the opponent's per-combo best-response value against the
// DEPLOYED (average) strategy must not exceed its alternative. alt comes
// from a plain solve of the same situation (the play-time recipe). A
// violation means the safety property doesn't hold and play-time use is
// unsound.
int run_gadget_check(int T) {
    PokerEnvironment env(poker_ppo::kPokerConfig,
                         poker_ppo::config::kBetConfig, 42);
    const int bb = env.game_config().big_blind;
    const std::vector<int> allowed = {0, 1, 4, 7, 13};
    double worst = 0.0, mean = 0.0;
    long cnt = 0;
    for (int trial = 0; trial < 5; ++trial) {
        advance_to_round(env, 3);
        if (env.is_terminal()) continue;
        std::mt19937 rng(100 + trial);
        HunlPBS beta;
        beta.r0 = random_range(rng);
        beta.r1 = random_range(rng);

        HunlSolver plain(env, beta, nullptr, allowed);
        for (int t = 1; t <= T; ++t) plain.iterate(t);
        std::array<std::vector<double>, 2> v, m;
        plain.values_at(0, v, m);
        const int opp = trial % 2;

        HunlSolver gd(env, beta, nullptr, allowed);
        gd.enable_gadget(opp, v[static_cast<size_t>(opp)], 0.1);
        for (int t = 1; t <= T; ++t) gd.iterate(t);

        // opponent best-responds to the gadgeted average strategy
        std::vector<double> us = opp == 0 ? beta.r1 : beta.r0;
        {   // normalize over valid (mirror the solver's own convention)
            std::vector<uint8_t> vmask;
            uint8_t b5[5];
            for (int i = 0; i < 5; ++i)
                b5[i] = static_cast<uint8_t>(env.community_card(i));
            board_valid(b5, 5, vmask);
            double s = 0.0;
            for (int i = 0; i < kCombos; ++i) {
                if (!vmask[i]) us[i] = 0.0;
                s += us[i];
            }
            for (double& x : us) x /= s > 0 ? s : 1.0;
            auto br = gd.best_response(opp, us);
            std::vector<double> mass;
            compat_mass(us, vmask, mass);
            for (int j = 0; j < kCombos; ++j) {
                if (!vmask[j] || mass[j] <= 1e-6 ||
                    m[static_cast<size_t>(opp)][j] < 0.5)
                    continue;
                const double viol =
                    br[j] / mass[j] - v[static_cast<size_t>(opp)][j];
                if (viol > worst) worst = viol;
                mean += std::max(0.0, viol);
                ++cnt;
            }
        }
    }
    mean = cnt > 0 ? mean / cnt : 0.0;
    // Finite-T CFR leaves margin slack that shrinks with T (measured:
    // max 0.68 bb @T=100, 0.154 @400, 0.034 @1600 — clean convergence to
    // the guarantee). The bar tracks that curve with ~2× headroom; a real
    // implementation bug shows up as O(pot) violations that do NOT shrink.
    const double bar = 8.0 * bb / std::sqrt(static_cast<double>(T));
    std::printf("gadget_check: T=%d combos=%ld  BR-over-alt violation "
                "mean=%.4f bb  max=%.4f bb  %s\n",
                T, cnt, mean / bb, worst / bb,
                worst < bar ? "PASS" : "FAIL");
    return worst < bar ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "kernels";
    if (mode == "kernels") return run_kernels();
    if (mode == "gadget_check")
        return run_gadget_check(argc > 2 ? std::atoi(argv[2]) : 400);
    if (mode == "gpu_check")
        return run_gpu_check(argc > 2 ? std::atoi(argv[2]) : 16,
                             argc > 3 ? std::atoi(argv[3]) : 200);
    if (mode == "gpu_bench")
        return run_gpu_bench(argc > 2 ? std::atoi(argv[2]) : 512,
                             argc > 3 ? std::atoi(argv[3]) : 200,
                             argc > 4 ? std::atoi(argv[4]) : 20);
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
    // Phase B′: LBR judges the re-solving agent — the same judge (and
    // config) the PPO 1.85 / ESCHER 1.256 numbers came from.
    //   rebel_hunl lbr <hands> <threads>
    //   REBEL_CKPT       hand-strength value net (default rebel_hs_turn.pt)
    //   REBEL_HIDDEN/LAYERS/GELU_LN/ZERO_SUM   net architecture
    //   REBEL_BLUEPRINT  PPO ActorCritic .pt for preflop/flop (hybrid);
    //                    empty → check/call stub (pipeline smoke ONLY)
    //   REBEL_T_TURN / REBEL_T_RIVER   play-time solve iterations
    if (mode == "lbr") {
        const int hands   = argc > 2 ? std::atoi(argv[2]) : 200;
        const int threads = argc > 3 ? std::atoi(argv[3]) : 1;
        const torch::Device device =
            torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        auto env_i = [](const char* n, int d) {
            const char* s = std::getenv(n);
            return s ? std::atoi(s) : d;
        };
        const int hidden  = env_i("REBEL_HIDDEN", 1024);
        const int layers  = env_i("REBEL_LAYERS", 2);
        const bool gelu   = env_i("REBEL_GELU_LN", 0) != 0;
        const bool zs     = env_i("REBEL_ZERO_SUM", 1) != 0;
        const char* ce    = std::getenv("REBEL_CKPT");
        const std::string ckpt = ce ? ce : "rebel_hs_turn.pt";

        HunlValueNet net(hidden, layers, gelu, zs);
        {   // shape-guarded load (torch::load replaces tensors silently)
            std::unordered_map<std::string, std::vector<int64_t>> want;
            for (const auto& p : net->named_parameters())
                want[p.key()] = p.value().sizes().vec();
            bool ok = true;
            try {
                torch::load(net, ckpt);
                for (const auto& p : net->named_parameters())
                    if (want.at(p.key()) != p.value().sizes().vec())
                        ok = false;
            } catch (const std::exception&) { ok = false; }
            if (!ok) {
                std::fprintf(stderr, "lbr: %s does not fit hidden=%d "
                             "layers=%d gelu_ln=%d\n", ckpt.c_str(), hidden,
                             layers, gelu ? 1 : 0);
                return 1;
            }
        }
        net->to(device);
        net->eval();

        poker_ppo::PokerEnvironmentFactory factory(poker_ppo::kPokerConfig);
        const auto& bet_cfg = poker_ppo::config::kBetConfig;
        auto probe_env = factory.create(bet_cfg);
        const int D = probe_env->obs_dim();
        const int A = bet_cfg.action_count();

        poker_ppo::LBRConfig lc;   // analytic mode, raises on: the
        lc.num_hands = hands;      // trustworthy-bound defaults
        if (const char* s = std::getenv("REBEL_SEED"))
            lc.seed = std::strtoull(s, nullptr, 10);
        // per-hand CSV (shards suffix .N) — aggregate attribution across
        // ALL shards, not just shard 0's printed table
        if (const char* s = std::getenv("REBEL_LBR_LOG")) lc.log_path = s;

        poker_ppo::ActorCritic bp{nullptr};
        std::unique_ptr<poker_ppo::ILBRTarget> bp_target;
        if (const char* b = std::getenv("REBEL_BLUEPRINT")) {
            bp = poker_ppo::ActorCritic(
                D, A, poker_ppo::config::kPPOConfig.hidden_dim,
                poker_ppo::config::kPPOConfig.num_layers,
                poker_ppo::config::kPPOConfig.hist,
                poker_ppo::config::kPPOConfig.round_summary);
            try {
                torch::load(bp, b);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "lbr: cannot load blueprint %s (%s)\n",
                             b, e.what());
                return 1;
            }
            bp->to(device);
            bp_target = poker_ppo::make_actor_critic_target(bp, lc, device,
                                                            D, A);
            std::printf("lbr: blueprint %s (preflop/flop)\n", b);
        } else {
            std::fprintf(stderr,
                         "lbr: [warn] no REBEL_BLUEPRINT — preflop/flop is a "
                         "check/call stub; the bound measures that toy "
                         "hybrid, not ReBeL\n");
        }

        // paired baseline: judge the BLUEPRINT alone under the exact same
        // binary/config/seed — the only comparison that isolates what the
        // re-solving layer adds (historical numbers may differ in config)
        if (env_i("REBEL_BLUEPRINT_ONLY", 0) != 0) {
            if (!bp_target) {
                std::fprintf(stderr, "lbr: BLUEPRINT_ONLY needs "
                             "REBEL_BLUEPRINT\n");
                return 1;
            }
            std::printf("lbr: judging the BLUEPRINT ALONE (no re-solving)\n");
            const auto res = poker_ppo::LBREvaluator::evaluate_sharded_target(
                factory, bet_cfg, lc, device,
                [&](int) {
                    return poker_ppo::make_actor_critic_target(bp, lc,
                                                               device, D, A);
                },
                threads);
            std::printf("\nLBR vs blueprint alone: %.4f bb/hand over %d "
                        "hands (LBR win rate %.3f, %.0fs)\n",
                        res.bb_per_hand, res.num_hands, res.lbr_win_rate,
                        res.wall_ms / 1000.0);
            return 0;
        }

        RebelPlayConfig pc;
        pc.t_turn  = env_i("REBEL_T_TURN", 120);
        pc.t_river = env_i("REBEL_T_RIVER", 200);
        pc.t_flop  = env_i("REBEL_T_FLOP", 60);
        pc.flop_solve = env_i("REBEL_FLOP", 1) != 0;
        // preflop re-solving (sampled-flop leaves valued by the net) —
        // OFF until a street-1-trained net exists; REBEL_PREFLOP=1 to try
        pc.preflop_solve = env_i("REBEL_PREFLOP", 0) != 0;
        pc.t_preflop     = env_i("REBEL_T_PREFLOP", 40);
        pc.pf_samples    = env_i("REBEL_PF_SAMPLES", 64);
        if (pc.preflop_solve)
            std::printf("lbr: PREFLOP re-solving ON (T=%d, %d sampled "
                        "flops/leaf)\n", pc.t_preflop, pc.pf_samples);
        pc.pcfr = env_i("REBEL_PCFR", 0) != 0;
        if (pc.pcfr) std::printf("lbr: PCFR+ ON (all play-time solves)\n");
        pc.gpu_turn = env_i("REBEL_GPU_TURN", 0) != 0;
        if (pc.gpu_turn)
            std::printf("lbr: GPU TURN solves ON (B=1 state transfer)\n");
        // default OFF: paired 20k A/B measured a wash (1.909 off vs 1.963
        // on, seed 1234) with a worse fold profile — the net-priced
        // alternatives run generous, combos terminate, the follow-range
        // skews strong and the solve over-folds. Sound machinery
        // (gadget_check converges), needs alt calibration to earn ON.
        pc.gadget  = env_i("REBEL_GADGET", 0) != 0;
        std::printf("lbr: river gadget %s\n", pc.gadget ? "ON" : "off");
        const double stack = static_cast<double>(
            poker_ppo::kPokerConfig.game.initial_stack);
        std::printf("lbr: hands=%d threads=%d net=%s (%dx%d%s) T=%d/%d "
                    "device=%s\n", hands, threads, ckpt.c_str(), hidden,
                    layers, gelu ? " gelu+ln" : "", pc.t_turn, pc.t_river,
                    device.is_cuda() ? "cuda" : "cpu");
        {   // HandRanks pre-warm: lazy load isn't thread-safe
            const uint8_t warm[5] = {0, 5, 10, 15, 20};
            (void)combo_rank(ComboTable::get().id[25][30], warm);
        }
        if (threads > 1) torch::set_num_threads(1);

        const auto res = poker_ppo::LBREvaluator::evaluate_sharded_target(
            factory, bet_cfg, lc, device,
            [&](int t) -> std::unique_ptr<poker_ppo::ILBRTarget> {
                RebelPlayConfig c = pc;
                c.seed = lc.seed + 104729u * static_cast<uint64_t>(t) + 13u;
                return std::make_unique<RebelTarget>(net, stack, device, c,
                                                     bp_target.get());
            },
            threads);
        std::printf("\nLBR vs ReBeL agent: %.4f bb/hand over %d hands "
                    "(LBR win rate %.3f, %.0fs)\n",
                    res.bb_per_hand, res.num_hands, res.lbr_win_rate,
                    res.wall_ms / 1000.0);
        return 0;
    }
    if (mode == "turn_gpu_check") {
        // BatchTurnSolver vs CPU HunlSolver on B lockstep turn subgames
        // with NET leaves, both fed by HunlNetOracle::value_batch (feature
        // and net parity by construction — only tree math differs).
        //   phase A: leaves frozen after t=1 → tight arithmetic bars
        //   phase B: live refresh every 5 → RM+ clamp order drift compounds
        //            through leaf-belief feedback; judge by the MEAN
        // usage: turn_gpu_check [B] [T]   (REBEL_CKPT required;
        //        REBEL_PCFR=1 checks the PCFR+ variant on both sides)
        const int B = argc > 2 ? std::atoi(argv[2]) : 4;
        const int T = argc > 3 ? std::atoi(argv[3]) : 40;
        const char* ck = std::getenv("REBEL_CKPT");
        const bool pcfr = std::getenv("REBEL_PCFR") &&
                          std::atoi(std::getenv("REBEL_PCFR")) != 0;
        if (!ck || !*ck) {
            std::fprintf(stderr, "turn_gpu_check needs REBEL_CKPT\n");
            return 1;
        }
        {
            const uint8_t warm[5] = {0, 5, 10, 15, 20};
            (void)combo_rank(ComboTable::get().id[25][30], warm);
        }
        HunlValueNet net(1024, 2, false, true);
        torch::load(net, ck);
        torch::Device dev(torch::cuda::is_available() ? torch::kCUDA
                                                      : torch::kCPU);
        net->to(dev);
        const double stack = static_cast<double>(
            poker_ppo::kPokerConfig.game.initial_stack);
        const std::vector<int> acts = {0, 1, 7, 13};
        std::mt19937_64 rng(20260715);
        std::uniform_real_distribution<double> u(0.0, 1.0);

        // collect B envs at turn roots sharing ONE tree signature
        std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>> envs;
        std::vector<TurnSpec> specs;
        std::vector<std::unique_ptr<HunlSolver>> cpus;   // shape source
        std::string sig;
        HunlNetOracle shape_oracle(net, stack, dev);
        for (int tries = 0; tries < 400 && static_cast<int>(envs.size()) < B;
             ++tries) {
            auto env = std::make_unique<poker_ppo::PokerEnvironment>(
                poker_ppo::kPokerConfig, poker_ppo::config::kBetConfig,
                31337 + 7 * tries);
            std::mt19937 wrng(2222 + 3 * tries);
            std::uniform_real_distribution<double> uw(0.0, 1.0);
            env->reset();
            while (!env->is_terminal() && env->round() < 2) {
                int a = 1;
                if (uw(wrng) < 0.3) {
                    static const int kR[] = {4, 7, 10};
                    a = kR[wrng() % 3];
                }
                auto mask = env->legal_action_mask();
                auto ma = mask.accessor<float, 1>();
                env->step(ma[a] > 0.5f ? a : 1);
            }
            if (env->is_terminal() || env->round() != 2) continue;
            TurnSpec sp;
            for (int i = 0; i < 4; ++i)
                sp.board[static_cast<size_t>(i)] =
                    static_cast<uint8_t>(env->community_card(i));
            sp.r0.resize(kCombos);
            sp.r1.resize(kCombos);
            for (int i = 0; i < kCombos; ++i) {
                sp.r0[static_cast<size_t>(i)] = std::pow(u(rng), 3.0);
                sp.r1[static_cast<size_t>(i)] = std::pow(u(rng), 3.0);
            }
            HunlPBS beta;
            beta.r0 = sp.r0;
            beta.r1 = sp.r1;
            auto s = std::make_unique<HunlSolver>(*env, beta, &shape_oracle,
                                                  acts);
            auto sh = TreeShape::from(*s);
            if (sig.empty()) sig = sh.signature;
            if (sh.signature != sig) continue;
            sp.node_contrib0.resize(s->nodes().size());
            sp.node_contrib1.resize(s->nodes().size());
            for (size_t m = 0; m < s->nodes().size(); ++m) {
                sp.node_contrib0[m] = s->nodes()[m].contrib[0];
                sp.node_contrib1[m] = s->nodes()[m].contrib[1];
            }
            envs.push_back(std::move(env));
            specs.push_back(std::move(sp));
            cpus.push_back(std::move(s));
        }
        if (static_cast<int>(envs.size()) < B) {
            std::fprintf(stderr, "only %zu/%d specs share a topology\n",
                         envs.size(), B);
            return 1;
        }
        auto shape = TreeShape::from(*cpus[0]);
        int n_leaves = 0;
        for (const auto& nd : shape.nodes)
            if (nd.kind == HunlSolver::Node::StreetEnd) ++n_leaves;
        std::printf("turn_gpu_check: B=%d T=%d nodes=%zu leaves=%d "
                    "pcfr=%d device=%s dtype=f64\n", B, T,
                    shape.nodes.size(), n_leaves, pcfr ? 1 : 0,
                    dev.is_cuda() ? "cuda" : "cpu");

        for (int phase = 0; phase < 3; ++phase) {
            // 0: f64 frozen leaves; 1: f64 live refresh; 2: f32 live
            const int refresh = phase == 0 ? 1000000 : 5;
            const auto gdt = phase == 2 ? torch::kFloat : torch::kDouble;
            // ── CPU side: fresh solvers, tracked root values ──
            std::vector<std::unique_ptr<HunlSolver>> cpu;
            std::vector<std::unique_ptr<HunlNetOracle>> oracles;
            for (int b = 0; b < B; ++b) {
                HunlPBS beta;
                beta.r0 = specs[static_cast<size_t>(b)].r0;
                beta.r1 = specs[static_cast<size_t>(b)].r1;
                oracles.push_back(std::make_unique<HunlNetOracle>(
                    net, stack, dev));
                auto s = std::make_unique<HunlSolver>(
                    *envs[static_cast<size_t>(b)], beta,
                    oracles.back().get(), acts);
                s->refresh_every = refresh;
                s->pcfr = pcfr;
                s->track_root_values();
                cpu.push_back(std::move(s));
            }
            for (int t = 1; t <= T; ++t)
                for (int b = 0; b < B; ++b) cpu[static_cast<size_t>(b)]
                    ->iterate(t);

            // ── GPU side: one batch; the shared oracle round-trip (pot
            // read off spec contribs — validates value_batch_pot against
            // the CPU side's env-based path) ──
            BatchTurnSolver gpu(shape, specs, dev, gdt, pcfr);
            std::vector<std::unique_ptr<HunlNetOracle>> gora;
            for (int b = 0; b < B; ++b)
                gora.push_back(std::make_unique<HunlNetOracle>(net, stack,
                                                               dev));
            std::vector<HunlNetOracle*> gop;
            for (auto& o : gora) gop.push_back(o.get());
            TurnRefreshWorkspace wsp;
            const bool dev_feat = gpu.dtype() == torch::kFloat &&
                                  !std::getenv("REBEL_NO_DEV_FEAT");
            gpu.solve(T, refresh, [&](int) {
                if (dev_feat)
                    gpu.refresh_leaves_device(net, stack);
                else
                    refresh_turn_leaves(gpu, specs, gop, 0, &wsp);
            });

            // ── compare iterate-averaged root values ──
            std::vector<std::array<std::vector<double>, 2>> gv, gm;
            gpu.avg_root_values(gv, gm);
            double mx = 0.0, mean = 0.0;
            long cnt = 0;
            for (int b = 0; b < B; ++b) {
                std::array<std::vector<double>, 2> cv, cm;
                cpu[static_cast<size_t>(b)]->avg_root_values(cv, cm);
                const double pot = static_cast<double>(
                    envs[static_cast<size_t>(b)]->pot());
                for (int p = 0; p < 2; ++p)
                    for (int x = 0; x < kCombos; ++x) {
                        if (cm[static_cast<size_t>(p)]
                              [static_cast<size_t>(x)] < 0.5 ||
                            gm[static_cast<size_t>(b)]
                              [static_cast<size_t>(p)]
                              [static_cast<size_t>(x)] < 0.5)
                            continue;
                        const double d = std::abs(
                            cv[static_cast<size_t>(p)]
                              [static_cast<size_t>(x)] -
                            gv[static_cast<size_t>(b)]
                              [static_cast<size_t>(p)]
                              [static_cast<size_t>(x)]) / pot;
                        mx = std::max(mx, d);
                        mean += d;
                        ++cnt;
                    }
            }
            mean = cnt ? mean / static_cast<double>(cnt) : 0.0;
            // f32 bars follow the river solver's calibration (mean =
            // arithmetic honesty; max = RM+ clamp-order equilibrium drift)
            const double bar_mean = phase == 0 ? 1e-6
                                  : phase == 1 ? 1e-3 : 2e-3;
            const double bar_max = phase == 0 ? 1e-4
                                 : phase == 1 ? 5e-2 : 5e-2;
            static const char* kPhase[] = {"A (f64 frozen leaves)",
                                           "B (f64 live refresh)",
                                           "C (f32 live refresh)"};
            std::printf("  phase %s: |dv|/pot mean=%.3e max=%.3e  %s\n",
                        kPhase[phase], mean, mx,
                        (mean <= bar_mean && mx <= bar_max) ? "PASS"
                                                            : "FAIL");
            std::fflush(stdout);
        }
        return 0;
    }
    if (mode == "turn_gpu_bench") {
        // Throughput: one batched GPU turn solve (graphed iterations +
        // oracle refreshes) vs the CPU solver on the same specs.
        // usage: turn_gpu_bench [B] [T]  (REBEL_CKPT required;
        //        REBEL_PCFR, REBEL_REFRESH=5, REBEL_CPU_REF=2)
        const int B = argc > 2 ? std::atoi(argv[2]) : 32;
        const int T = argc > 3 ? std::atoi(argv[3]) : 120;
        const int refresh = std::getenv("REBEL_REFRESH")
            ? std::atoi(std::getenv("REBEL_REFRESH")) : 5;
        int cpu_ref = std::getenv("REBEL_CPU_REF")
            ? std::atoi(std::getenv("REBEL_CPU_REF")) : 2;
        const char* ck = std::getenv("REBEL_CKPT");
        const bool pcfr = std::getenv("REBEL_PCFR") &&
                          std::atoi(std::getenv("REBEL_PCFR")) != 0;
        if (!ck || !*ck) {
            std::fprintf(stderr, "turn_gpu_bench needs REBEL_CKPT\n");
            return 1;
        }
        {
            const uint8_t warm[5] = {0, 5, 10, 15, 20};
            (void)combo_rank(ComboTable::get().id[25][30], warm);
        }
        HunlValueNet net(1024, 2, false, true);
        torch::load(net, ck);
        torch::Device dev(torch::cuda::is_available() ? torch::kCUDA
                                                      : torch::kCPU);
        net->to(dev);
        const double stack = static_cast<double>(
            poker_ppo::kPokerConfig.game.initial_stack);
        const std::vector<int> acts = {0, 1, 7, 13};
        std::mt19937_64 rng(20260716);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        HunlNetOracle shape_oracle(net, stack, dev);
        std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>> envs;
        std::vector<TurnSpec> specs;
        std::string sig;
        TreeShape shape;
        for (int tries = 0; tries < std::max(4000, 300 * B) &&
                            static_cast<int>(specs.size()) < B; ++tries) {
            auto env = std::make_unique<poker_ppo::PokerEnvironment>(
                poker_ppo::kPokerConfig, poker_ppo::config::kBetConfig,
                60601 + 11 * tries);
            std::mt19937 wrng(881 + 5 * tries);
            std::uniform_real_distribution<double> uw(0.0, 1.0);
            env->reset();
            while (!env->is_terminal() && env->round() < 2) {
                int a = 1;
                if (uw(wrng) < 0.3) {
                    static const int kR[] = {4, 7, 10};
                    a = kR[wrng() % 3];
                }
                auto mask = env->legal_action_mask();
                auto ma = mask.accessor<float, 1>();
                env->step(ma[a] > 0.5f ? a : 1);
            }
            if (env->is_terminal() || env->round() != 2) continue;
            TurnSpec sp;
            for (int i = 0; i < 4; ++i)
                sp.board[static_cast<size_t>(i)] =
                    static_cast<uint8_t>(env->community_card(i));
            sp.r0.resize(kCombos);
            sp.r1.resize(kCombos);
            for (int i = 0; i < kCombos; ++i) {
                sp.r0[static_cast<size_t>(i)] = std::pow(u(rng), 3.0);
                sp.r1[static_cast<size_t>(i)] = std::pow(u(rng), 3.0);
            }
            HunlPBS beta;
            beta.r0 = sp.r0;
            beta.r1 = sp.r1;
            HunlSolver s(*env, beta, &shape_oracle, acts);
            auto sh = TreeShape::from(s);
            if (sig.empty()) {
                sig = sh.signature;
                shape = sh;
            }
            if (sh.signature != sig) continue;
            sp.node_contrib0.resize(s.nodes().size());
            sp.node_contrib1.resize(s.nodes().size());
            for (size_t m = 0; m < s.nodes().size(); ++m) {
                sp.node_contrib0[m] = s.nodes()[m].contrib[0];
                sp.node_contrib1[m] = s.nodes()[m].contrib[1];
            }
            envs.push_back(std::move(env));
            specs.push_back(std::move(sp));
        }
        if (static_cast<int>(specs.size()) < B) {
            std::fprintf(stderr, "only %zu/%d specs share a topology\n",
                         specs.size(), B);
            return 1;
        }
        using clk = std::chrono::steady_clock;
        auto secs = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        std::printf("turn_gpu_bench: B=%d T=%d refresh=%d pcfr=%d nodes=%zu "
                    "device=%s f32\n", B, T, refresh, pcfr ? 1 : 0,
                    shape.nodes.size(), dev.is_cuda() ? "cuda" : "cpu");
        auto tc0 = clk::now();
        BatchTurnSolver gpu(shape, specs, dev, torch::kFloat, pcfr);
        auto tc1 = clk::now();
        std::vector<std::unique_ptr<HunlNetOracle>> gora;
        for (int b = 0; b < B; ++b)
            gora.push_back(std::make_unique<HunlNetOracle>(net, stack, dev));
        std::vector<HunlNetOracle*> gop;
        for (auto& o : gora) gop.push_back(o.get());
        double cb_s = 0.0;
        TurnRefreshWorkspace wsp;
        const bool dev_feat = !std::getenv("REBEL_NO_DEV_FEAT");
        auto ts0 = clk::now();
        gpu.solve(T, refresh, [&](int) {
            auto c0 = clk::now();
            if (dev_feat)
                gpu.refresh_leaves_device(net, stack);
            else
                refresh_turn_leaves(gpu, specs, gop, 0, &wsp);
            cb_s += secs(c0, clk::now());
        });
        std::vector<std::array<std::vector<double>, 2>> gv, gm;
        gpu.avg_root_values(gv, gm);
        auto ts1 = clk::now();
        const double wall = secs(ts0, ts1);
        std::printf("  gpu: ctor %.2fs  solve %.2fs (refresh %.2fs = %.0f%%)"
                    "  -> %.1f targets/s\n", secs(tc0, tc1), wall, cb_s,
                    100.0 * cb_s / wall, B / wall);
        cpu_ref = std::min(cpu_ref, B);
        if (cpu_ref > 0) {
            auto tr0 = clk::now();
            for (int b = 0; b < cpu_ref; ++b) {
                HunlPBS beta;
                beta.r0 = specs[static_cast<size_t>(b)].r0;
                beta.r1 = specs[static_cast<size_t>(b)].r1;
                HunlNetOracle orc(net, stack, dev);
                HunlSolver s(*envs[static_cast<size_t>(b)], beta, &orc,
                             acts);
                s.refresh_every = refresh;
                s.pcfr = pcfr;
                s.track_root_values();
                for (int t = 1; t <= T; ++t) s.iterate(t);
                std::array<std::vector<double>, 2> v, m;
                s.avg_root_values(v, m);
            }
            const double cpu_s = secs(tr0, clk::now()) / cpu_ref;
            std::printf("  cpu: %.2fs/solve (1 thread)  -> batch speedup "
                        "%.1fx (vs %d such threads: %.1fx)\n", cpu_s,
                        cpu_s * B / wall, B, cpu_s / wall);
        }
        return 0;
    }
    if (mode == "pcfr_bench") {
        // CFR+ vs PCFR+ on K fixed exact river subgames:
        //  (1) exploitability vs iterations — the iteration-reduction factor
        //  (2) root-value agreement of PCFR+ at reduced T vs a LONG CFR+
        //      reference, judged against CFR+'s own self-distance — the
        //      "doesn't affect training targets" check.
        const int K = argc > 2 ? std::atoi(argv[2]) : 3;
        {
            const uint8_t warm[5] = {0, 5, 10, 15, 20};
            (void)combo_rank(ComboTable::get().id[25][30], warm);
        }
        const double bb = static_cast<double>(
            poker_ppo::kPokerConfig.game.big_blind);
        const std::vector<int> acts = {0, 1, 4, 7, 13};
        static const int kT[] = {50, 100, 200, 400, 800};
        std::mt19937_64 rng(20260714);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int k = 0; k < K; ++k) {
            poker_ppo::PokerEnvironment env(poker_ppo::kPokerConfig,
                                            poker_ppo::config::kBetConfig,
                                            777 + 131 * k);
            std::mt19937 wrng(555 + 31 * k);
            std::uniform_real_distribution<double> uw(0.0, 1.0);
            for (int tries = 0; tries < 50; ++tries) {
                env.reset();
                while (!env.is_terminal() && env.round() < 3) {
                    int a = 1;
                    if (uw(wrng) < 0.3) {
                        static const int kR[] = {4, 7, 10};
                        a = kR[wrng() % 3];
                    }
                    auto mask = env.legal_action_mask();
                    auto ma = mask.accessor<float, 1>();
                    env.step(ma[a] > 0.5f ? a : 1);
                }
                if (!env.is_terminal() && env.round() == 3) break;
            }
            if (env.is_terminal() || env.round() != 3) continue;
            HunlPBS beta;
            beta.r0.resize(kCombos);
            beta.r1.resize(kCombos);
            for (int i = 0; i < kCombos; ++i) {
                beta.r0[i] = std::pow(u(rng), 3.0);
                beta.r1[i] = std::pow(u(rng), 3.0);
            }
            std::printf("[situation %d] pot=%d\n", k, env.pot());
            for (int T : kT) {
                double e[3];
                for (int variant = 0; variant < 3; ++variant) {
                    HunlSolver s(env, beta, nullptr, acts);
                    s.pcfr = variant >= 1;
                    s.pcfr_quad = variant == 1;
                    for (int t = 1; t <= T; ++t) s.iterate(t);
                    e[variant] = s.exploitability() / bb;
                }
                std::printf("  T=%4d  cfr+ %.5f   pcfr+q %.5f (%.1fx)   "
                            "pcfr+lin %.5f (%.1fx) bb\n", T, e[0],
                            e[1], e[1] > 0 ? e[0] / e[1] : 0.0,
                            e[2], e[2] > 0 ? e[0] / e[2] : 0.0);
                std::fflush(stdout);
            }
            // (2) root-value agreement, first situation only (it's the
            // expensive part): reference = CFR+ T=3200
            if (k == 0) {
                auto vals = [&](bool pcfr, int T) {
                    HunlSolver s(env, beta, nullptr, acts);
                    s.pcfr = pcfr;
                    for (int t = 1; t <= T; ++t) s.iterate(t);
                    std::array<std::vector<double>, 2> v, m;
                    s.root_values(v, m);
                    return std::make_pair(v, m);
                };
                auto [vr, mr] = vals(false, 3200);
                const double pot = static_cast<double>(env.pot());
                auto dist = [&](const std::array<std::vector<double>, 2>& v,
                                const std::array<std::vector<double>, 2>& m) {
                    double s = 0.0;
                    long n = 0;
                    for (int p = 0; p < 2; ++p)
                        for (int i = 0; i < kCombos; ++i)
                            if (m[p][i] > 0.5 && mr[p][i] > 0.5) {
                                s += std::abs(v[p][i] - vr[p][i]) / pot;
                                ++n;
                            }
                    return n ? s / n : 0.0;
                };
                for (int T : {200, 400, 800}) {
                    auto [vc, mc] = vals(false, T);
                    auto [vp, mp] = vals(true, T);
                    std::printf("  root-value mean|d|/pot vs cfr+@3200:  "
                                "cfr+@%d %.2e   pcfr+@%d %.2e\n",
                                T, dist(vc, mc), T, dist(vp, mp));
                    std::fflush(stdout);
                }
            }
        }
        // ── turn subgames with NET leaves (the dominant play/training
        // cost) — needs REBEL_CKPT. exploitability() here is BR within
        // the leaf-valued game: comparative, not absolute.
        const char* ck = std::getenv("REBEL_CKPT");
        if (ck && *ck) {
            HunlValueNet net(1024, 2, false, true);
            torch::load(net, ck);
            torch::Device dev(torch::cuda::is_available() ? torch::kCUDA
                                                          : torch::kCPU);
            net->to(dev);
            const double stack = static_cast<double>(
                poker_ppo::kPokerConfig.game.initial_stack);
            const std::vector<int> tacts = {0, 1, 7, 13};
            for (int k = 0; k < K; ++k) {
                poker_ppo::PokerEnvironment env(
                    poker_ppo::kPokerConfig, poker_ppo::config::kBetConfig,
                    999 + 17 * k);
                std::mt19937 wrng(444 + 13 * k);
                std::uniform_real_distribution<double> uw(0.0, 1.0);
                for (int tries = 0; tries < 50; ++tries) {
                    env.reset();
                    while (!env.is_terminal() && env.round() < 2) {
                        int a = 1;
                        if (uw(wrng) < 0.3) {
                            static const int kR[] = {4, 7, 10};
                            a = kR[wrng() % 3];
                        }
                        auto mask = env.legal_action_mask();
                        auto ma = mask.accessor<float, 1>();
                        env.step(ma[a] > 0.5f ? a : 1);
                    }
                    if (!env.is_terminal() && env.round() == 2) break;
                }
                if (env.is_terminal() || env.round() != 2) continue;
                HunlPBS beta;
                beta.r0.resize(kCombos);
                beta.r1.resize(kCombos);
                for (int i = 0; i < kCombos; ++i) {
                    beta.r0[i] = std::pow(u(rng), 3.0);
                    beta.r1[i] = std::pow(u(rng), 3.0);
                }
                std::printf("[turn situation %d] pot=%d (net leaves)\n", k,
                            env.pot());
                for (int T : {30, 60, 120, 240}) {
                    double e[3];
                    for (int variant = 0; variant < 3; ++variant) {
                        HunlNetOracle oracle(net, stack, dev);
                        HunlSolver s(env, beta, &oracle, tacts);
                        s.refresh_every = 5;
                        s.pcfr = variant >= 1;
                        s.pcfr_quad = variant == 1;
                        for (int t = 1; t <= T; ++t) s.iterate(t);
                        e[variant] = s.exploitability() / bb;
                    }
                    std::printf("  T=%4d  cfr+ %.5f   pcfr+q %.5f (%.1fx)   "
                                "pcfr+lin %.5f (%.1fx) bb\n", T, e[0],
                                e[1], e[1] > 0 ? e[0] / e[1] : 0.0,
                                e[2], e[2] > 0 ? e[0] / e[2] : 0.0);
                    std::fflush(stdout);
                }
            }
        } else {
            std::printf("(set REBEL_CKPT for the net-leaf turn section)\n");
        }
        return 0;
    }
    if (mode == "flopctx_check") {
        // Self-consistency guard for the preflop-solve feature cache:
        // HunlNetOracle::flop_features (cached evaluators) must be
        // BIT-EXACT vs HunlFeaturizer::features (training-row path), and
        // both must be invariant to board ordering (training rows see the
        // env's deal order; preflop leaves see pf_flops_'s shuffle order).
        const int K = argc > 2 ? std::atoi(argv[2]) : 20;
        const double stack = static_cast<double>(
            poker_ppo::kPokerConfig.game.initial_stack);
        {
            const uint8_t warm[5] = {0, 5, 10, 15, 20};
            (void)combo_rank(ComboTable::get().id[25][30], warm);
        }
        HunlValueNet net(64, 2, false, true);   // never forwarded
        HunlNetOracle oracle(net, stack, torch::kCPU);
        std::mt19937_64 rng(4242);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        float maxd = 0.0f, maxp = 0.0f;
        for (int k = 0; k < K; ++k) {
            uint8_t deck[52];
            for (int c = 0; c < 52; ++c) deck[c] = static_cast<uint8_t>(c);
            for (int j = 0; j < 3; ++j)
                std::swap(deck[j], deck[j + rng() % (52 - j)]);
            const uint8_t flop[3] = {deck[0], deck[1], deck[2]};
            const uint8_t perm[3] = {deck[2], deck[0], deck[1]};
            HunlPBS beta;
            beta.r0.resize(kCombos);
            beta.r1.resize(kCombos);
            for (int i = 0; i < kCombos; ++i) {
                beta.r0[i] = std::pow(u(rng), 3.0);
                beta.r1[i] = std::pow(u(rng), 3.0);
            }
            const double pot = 2000.0 + static_cast<double>(rng() % 40000);
            const auto f1 =
                HunlFeaturizer::features(1, flop, 3, pot, stack, beta);
            const auto f2 = oracle.flop_features(flop, pot, beta);  // build
            const auto f3 = oracle.flop_features(flop, pot, beta);  // hit
            const auto f4 =
                HunlFeaturizer::features(1, perm, 3, pot, stack, beta);
            const auto f5 = oracle.flop_features(perm, pot, beta);
            for (int i = 0; i < HunlFeaturizer::kDim; ++i) {
                maxd = std::max(maxd, std::abs(f1[i] - f2[i]));
                maxd = std::max(maxd, std::abs(f1[i] - f3[i]));
                maxp = std::max(maxp, std::abs(f1[i] - f4[i]));
                maxp = std::max(maxp, std::abs(f1[i] - f5[i]));
            }
        }
        std::printf("flopctx_check: K=%d  cached-vs-uncached max|dF|=%g  "
                    "order-invariance max|dF|=%g  %s\n",
                    K, maxd, maxp,
                    (maxd == 0.0f && maxp == 0.0f) ? "PASS (bit-exact)"
                                                   : "FAIL");
        return (maxd == 0.0f && maxp == 0.0f) ? 0 : 1;
    }
    if (mode == "convert_data") {
        if (argc < 4) {
            std::fprintf(stderr, "usage: rebel_hunl convert_data <in> <out>\n");
            return 1;
        }
        return convert_dataset(argv[2], argv[3]);
    }
    if (mode == "train_turn" || mode == "train_river" ||
        mode == "train_flop" || mode == "train_root") {
        EndgameConfig cfg;
        cfg.river_only = (mode == "train_river");
        cfg.flop_mode = (mode == "train_flop");
        cfg.root_mode = (mode == "train_root");
        if (cfg.river_only) {
            cfg.episodes = 2000;   // direct river targets per epoch
            cfg.sgd_steps = 500;
        }
        if (cfg.flop_mode) {
            cfg.episodes = 50;   // a flop episode ≈ 1 flop + ~3 turn solves
            cfg.harvest = 2;
        }
        if (cfg.root_mode) {
            cfg.episodes = 30;   // ≈ 1 preflop + ~3 flop continuations,
            cfg.harvest = 2;     //   each ≈ 1 flop + ~3 turn solves
        }
        if (argc > 2) cfg.epochs = std::atoi(argv[2]);
        if (argc > 3) cfg.episodes = std::atoi(argv[3]);
        auto env_int = [](const char* name, int& v) {
            if (const char* s = std::getenv(name)) v = std::atoi(s);
        };
        env_int("REBEL_T_FLOP", cfg.t_flop);
        env_int("REBEL_T_TURN_TRAIN", cfg.t_turn);
        env_int("REBEL_T_PREFLOP", cfg.t_preflop);
        env_int("REBEL_PF_SAMPLES", cfg.pf_samples);
        env_int("REBEL_THREADS", cfg.threads);
        env_int("REBEL_GPU_BATCH", cfg.gpu_batch);
        env_int("REBEL_HARVEST", cfg.harvest);
        env_int("REBEL_HIDDEN", cfg.hidden);
        env_int("REBEL_LAYERS", cfg.layers);
        env_int("REBEL_SGD_STEPS", cfg.sgd_steps);
        env_int("REBEL_BATCH", cfg.batch);
        env_int("REBEL_PROBE_K", cfg.probe_k);
        env_int("REBEL_REPLAY_CAP", cfg.replay_cap);
        if (const char* s = std::getenv("REBEL_LR")) cfg.lr = std::atof(s);
        if (const char* s = std::getenv("REBEL_LR_FINAL"))
            cfg.lr_final = std::atof(s);
        if (const char* s = std::getenv("REBEL_HUBER_DELTA"))
            cfg.huber_delta = std::atof(s);
        if (const char* s = std::getenv("REBEL_GELU_LN"))
            cfg.gelu_ln = std::atoi(s) != 0;
        if (const char* s = std::getenv("REBEL_ZERO_SUM"))
            cfg.zero_sum = std::atoi(s) != 0;
        if (const char* s = std::getenv("REBEL_PCFR"))
            cfg.pcfr = std::atoi(s) != 0;
        env_int("REBEL_CIRCULAR", cfg.circular);
        if (const char* s = std::getenv("REBEL_CKPT")) cfg.ckpt = s;
        if (const char* s = std::getenv("REBEL_DATA_IN")) cfg.data_in = s;
        if (const char* s = std::getenv("REBEL_DATA_OUT")) cfg.data_out = s;
        if (const char* s = std::getenv("REBEL_SEED"))
            cfg.seed = std::strtoull(s, nullptr, 10);
        EndgameTrainer tr(cfg);
        tr.run();
        return 0;
    }
    std::fprintf(stderr,
                 "mode must be kernels|river|turn|train_turn|train_river|"
                 "train_flop|lbr\n");
    return 1;
}
