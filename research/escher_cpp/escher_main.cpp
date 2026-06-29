// Neural ESCHER (libtorch C++) runner — validates on Kuhn / Leduc.
//   escher <kuhn|kuhn6|leduc> <exact|net> [iters] [eval_every]
//          [--neural-cum] [--ncum-gamma G] [--sampled] [--n-traj K]
//          [--val-sweeps S]
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <string>

#include "escher.h"
#include "games.h"

using namespace escher;

int main(int argc, char** argv) {
    std::string which = argc > 1 ? argv[1] : "kuhn6";
    std::string mode = argc > 2 ? argv[2] : "exact";
    int iters = 120, eval_every = 20;

    NeuralESCHER::Config cfg;
    cfg.mode = (mode == "net") ? NeuralESCHER::Mode::Net : NeuralESCHER::Mode::Exact;
    cfg.hidden = 128;
    cfg.reg_steps = 400;

    // positional iters/eval_every (if numeric), then named flags
    int pos = 3;
    if (argc > pos && argv[pos][0] != '-') iters = std::atoi(argv[pos++]);
    if (argc > pos && argv[pos][0] != '-') eval_every = std::atoi(argv[pos++]);
    for (int i = pos; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--neural-cum") cfg.neural_cum = true;
        else if (a == "--sampled") cfg.sampled = true;
        else if (a == "--ncum-gamma" && i + 1 < argc) cfg.ncum_gamma = std::atof(argv[++i]);
        else if (a == "--n-traj" && i + 1 < argc) cfg.n_traj = std::atoi(argv[++i]);
        else if (a == "--val-sweeps" && i + 1 < argc) cfg.val_sweeps = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }

    std::unique_ptr<Game> g;
    int n_ranks;
    if (which == "kuhn") { g = std::make_unique<Kuhn>(); n_ranks = 3; }
    else if (which == "kuhn6") { g = std::make_unique<KuhnN>(6); n_ranks = 6; }
    else if (which == "leduc") { g = std::make_unique<Leduc>(); n_ranks = 3; }
    else { std::fprintf(stderr, "game must be kuhn|kuhn6|leduc\n"); return 1; }

    std::printf("Neural ESCHER (C++) on %s (values=%s, hidden=%d, neural_cum=%d, "
                "sampled=%d, n_traj=%d, gamma=%.2f):\n",
                which.c_str(), mode.c_str(), cfg.hidden, cfg.neural_cum,
                cfg.sampled, cfg.n_traj, cfg.ncum_gamma);
    NeuralESCHER esc(*g, n_ranks, cfg);
    esc.run(iters, eval_every);
    return 0;
}
