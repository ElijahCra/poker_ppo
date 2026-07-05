// ReBeL research runner — validation ladder before any HUNL wiring:
//   rebel <kuhn|kuhn6|leduc> exact [T...]   solver + decomposition test:
//       agent strategy from solves with the EXACT recursive leaf oracle
//       (CFR-D). Exploitability should fall with T at ~CFR+ rates; this
//       validates the vectorized solver, chance handling, card removal and
//       value-normalization semantics with NO learning in the loop.
//   rebel <game> train [epochs]             full ReBeL (Algorithm 1) with
//       the learned value net; reports value loss, probe MSE vs the exact
//       oracle, and exact exploitability of the re-solving agent per epoch.
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "games.h"
#include "rebel.h"

int main(int argc, char** argv) {
    const std::string which = argc > 1 ? argv[1] : "leduc";
    const std::string mode = argc > 2 ? argv[2] : "exact";

    std::unique_ptr<escher::Game> g;
    rebel::DeckSpec deck;
    if (which == "kuhn") {
        g = std::make_unique<escher::Kuhn>();
        deck = rebel::kuhn_deck(3);
    } else if (which == "kuhn6") {
        g = std::make_unique<escher::KuhnN>(6);
        deck = rebel::kuhn_deck(6);
    } else if (which == "leduc") {
        g = std::make_unique<escher::Leduc>();
        deck = rebel::leduc_deck();
    } else {
        std::fprintf(stderr, "game must be kuhn|kuhn6|leduc\n");
        return 1;
    }

    rebel::Config cfg;
    if (std::getenv("REBEL_CFR_D") != nullptr) cfg.cfr_avg = false;
    if (const char* s = std::getenv("REBEL_EVAL_SMOOTH"))
        cfg.eval_smooth = std::atof(s);
    if (const char* s = std::getenv("REBEL_EXACT_ITERS"))
        cfg.exact_iters = std::atoi(s);
    if (mode == "exact") {
        rebel::Trainer tr(*g, deck, cfg);
        rebel::ExactOracle oracle(*g, deck, cfg.exact_iters);
        std::vector<int> ts = {50, 200, 1000};
        if (argc > 3) {
            ts.clear();
            for (int i = 3; i < argc; ++i) ts.push_back(std::atoi(argv[i]));
        }
        for (int t : ts) {
            auto strat = tr.agent_strategy(&oracle, t);
            std::printf("  CFR-D exact-leaf  T=%-5d expl=%.6f\n", t,
                        escher::exploitability(*g, strat));
            std::fflush(stdout);
        }
        return 0;
    }
    if (mode == "train") {
        if (argc > 3) cfg.epochs = std::atoi(argv[3]);
        rebel::Trainer tr(*g, deck, cfg);
        tr.run();
        return 0;
    }
    std::fprintf(stderr, "mode must be exact|train\n");
    return 1;
}
