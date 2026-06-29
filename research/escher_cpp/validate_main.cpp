// Phase-1 validation: confirm the C++ games + exact BR + CFR+ reproduce the
// validated Python numbers BEFORE adding the libtorch neural ESCHER.
//   Kuhn  CFR+ -> ~0.004
//   Leduc CFR+ -> ~0.05, game value ~ -0.0836 (matches literature -0.0856)
// Build (torch-free):
//   g++ -std=c++20 -O2 validate_main.cpp -o validate && ./validate
#include <cstdio>

#include "games.h"
#include "solvers.h"

using namespace escher;

static void run(const Game& g, int iters, int n_evals) {
    CFRPlus cfr(g);
    int last = 0;
    std::printf("== %s ==\n", g.name().c_str());
    for (int e = 1; e <= n_evals; ++e) {
        int it = iters * e / n_evals;
        cfr.iterate(it - last);
        last = it;
        auto avg = cfr.average_strategy();
        std::printf("  CFR+ @%-6d expl=%.6f  value=%+.5f\n", it,
                    exploitability(g, avg), expected_value(g, avg));
    }
}

int main() {
    Kuhn kuhn;
    run(kuhn, 2000, 4);
    KuhnN kuhn6(6);
    run(kuhn6, 2000, 4);
    Leduc leduc;
    run(leduc, 3000, 5);
    return 0;
}
