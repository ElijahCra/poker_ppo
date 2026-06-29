// Neural ESCHER (libtorch C++) runner — validates on Kuhn / Leduc.
//   escher <kuhn|kuhn6|leduc> <exact|net> [iters] [eval_every]
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

#include "escher.h"
#include "games.h"

using namespace escher;

int main(int argc, char** argv) {
    std::string which = argc > 1 ? argv[1] : "kuhn6";
    std::string mode = argc > 2 ? argv[2] : "exact";
    int iters = argc > 3 ? std::atoi(argv[3]) : 120;
    int eval_every = argc > 4 ? std::atoi(argv[4]) : 20;

    std::unique_ptr<Game> g;
    int n_ranks;
    if (which == "kuhn") { g = std::make_unique<Kuhn>(); n_ranks = 3; }
    else if (which == "kuhn6") { g = std::make_unique<KuhnN>(6); n_ranks = 6; }
    else if (which == "leduc") { g = std::make_unique<Leduc>(); n_ranks = 3; }
    else { std::fprintf(stderr, "game must be kuhn|kuhn6|leduc\n"); return 1; }

    NeuralESCHER::Config cfg;
    cfg.mode = (mode == "net") ? NeuralESCHER::Mode::Net : NeuralESCHER::Mode::Exact;
    cfg.hidden = 128;
    cfg.reg_steps = 400;

    std::printf("Neural ESCHER (C++) on %s (values=%s, hidden=%d):\n",
                which.c_str(), mode.c_str(), cfg.hidden);
    NeuralESCHER esc(*g, n_ranks, cfg);
    esc.run(iters, eval_every);
    return 0;
}
