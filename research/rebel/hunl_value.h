// HUNL PBS value net + Algorithm-1 endgame self-play (ReBeL stage 2).
//
// The net maps a street-root PBS to per-combo values for both players:
//   input  [street(4) | board multi-hot(52) | pot/stack | r0(1326) | r1(1326)]
//   output [v0(1326) | v1(1326)]   values normalized by the initial stack
// Trained on river-root targets generated the validated Leduc way: turn
// subgames are solved with NET leaves; at a uniformly sampled iteration t*
// the continuation leaf (river-root PBS) is sampled under the average
// profile with ε-exploration (Algorithm 1); that river subgame is solved
// EXACTLY (no further leaves) and its root values become the training
// target. The net is only ever queried at PBSs matching this distribution —
// the paper's self-consistency requirement.
//
// Turn-root sampling (DeepStack-style synthetic situations): random boards
// and pots via random legal env prefixes, random ranges — broad coverage of
// the river-root feature space beyond any single blueprint's play.
#pragma once

#include <torch/torch.h>

#include <array>
#include <memory>
#include <random>
#include <vector>

#include "hunl_solver.h"

namespace rebel_hunl {

class HunlFeaturizer {
public:
    static constexpr int kDim = 4 + kCards + 1 + 2 * kCombos;
    // contribs equal at a street root; pot = 2*contrib.
    static std::vector<float> features(int street, const uint8_t* board,
                                       int nb, double pot, double stack,
                                       const HunlPBS& beta);
};

struct HunlValueNetImpl : torch::nn::Module {
    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    HunlValueNetImpl(int hidden);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(HunlValueNet);

class HunlNetOracle : public HunlValueOracle {
public:
    HunlNetOracle(HunlValueNet net, double stack) : net_(net), stack_(stack) {}
    void value(poker_ppo::PokerEnvironment& env, const uint8_t* board, int nb,
               const HunlPBS& beta,
               std::array<std::vector<double>, 2>& out) override;

private:
    HunlValueNet net_;
    double stack_;
};

struct EndgameConfig {
    int    t_turn       = 120;   // CFR iterations per turn self-play solve
    int    t_river      = 200;   // exact river solve for targets
    int    episodes     = 32;    // per epoch
    int    epochs       = 20;
    int    sgd_steps    = 300;
    int    batch        = 128;
    int    hidden       = 1024;
    double lr           = 1e-3;
    double eps_explore  = 0.25;
    int    replay_cap   = 100000;
    int    probe_k      = 8;     // replay entries re-solved exactly per epoch
    std::vector<int> actions = {0, 1, 7, 13};   // sparse abstraction
    uint64_t seed       = 0;
};

class EndgameTrainer {
public:
    EndgameTrainer(EndgameConfig cfg);
    void run();

private:
    // env positioned at a random turn root (random board, random pot via a
    // random legal prefix); returns false if the prefix ended the hand.
    bool sample_turn_root(poker_ppo::PokerEnvironment& env, std::mt19937& rng);
    std::vector<double> random_range(std::mt19937& rng);
    void self_play_episode(poker_ppo::PokerEnvironment& env,
                           std::mt19937& rng);
    double train_net();
    // Masked MSE of the net vs stored targets on a random replay subset —
    // targets here ARE exact river solves, so this is a true accuracy probe
    // (unlike Leduc, no bootstrapped-target caveat).
    double probe_mse(int k);
    // Internal exploitability of a turn solve on a FIXED probe situation
    // with the given oracle — comparing net leaves vs exact leaves on the
    // same root is the cross-validation of the trained net.
    double probe_turn_expl(poker_ppo::PokerEnvironment& env,
                           HunlValueOracle* oracle, int T, int refresh_every);

    EndgameConfig cfg_;
    double        stack_;
    HunlValueNet  net_{nullptr};
    std::unique_ptr<torch::optim::Adam> opt_;
    std::mt19937  rng_;

    struct Sample {
        std::vector<float> feat, target, mask;   // target/mask [2*kCombos]
    };
    std::vector<Sample> replay_;
    long                seen_ = 0;
};

}  // namespace rebel_hunl
