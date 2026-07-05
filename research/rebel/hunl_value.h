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
    HunlNetOracle(HunlValueNet net, double stack, torch::Device device)
        : net_(net), stack_(stack), device_(device) {}
    void value(poker_ppo::PokerEnvironment& env, const uint8_t* board, int nb,
               const HunlPBS& beta,
               std::array<std::vector<double>, 2>& out) override;
    // one [N, kDim] forward for a whole leaf's candidate cards
    void value_batch(poker_ppo::PokerEnvironment& env,
                     const uint8_t* base_board, int nb_base,
                     const std::vector<uint8_t>& cards,
                     const std::vector<HunlPBS>& betas,
                     std::vector<std::array<std::vector<double>, 2>>& outs)
        override;

private:
    HunlValueNet net_;
    double stack_;
    torch::Device device_;
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
    // self-play worker threads (0 = hardware_concurrency). Episodes are
    // independent: each worker owns an env + RNG; the net is read-only at
    // inference. torch intra-op threads are pinned to 1 when workers > 1.
    int    threads      = 0;
    // Extra river targets harvested per turn solve from the (leaf, card)
    // PBSs the solver queries at final-average beliefs — the net's exact
    // query distribution, at ~1 river solve each. Algorithm 1's t* leaf
    // sample alone is 1 target per ~35 core-seconds: sample-starved (this
    // regression needs 1e5-1e7 rows; DeepStack used ~1M river situations).
    int    harvest      = 8;
    // train_river mode: direct river-situation sampling (DeepStack recipe) —
    // no turn solves at all; `episodes` = river targets per epoch.
    bool   river_only   = false;
    // net checkpoint: loaded at start if present, saved each epoch. Lets a
    // train_turn run fine-tune a river-pretrained net (and later serves the
    // play-time solver). Empty = off.
    std::string ckpt    = "rebel_value.pt";
    uint64_t seed       = 0;
};

class EndgameTrainer {
public:
    EndgameTrainer(EndgameConfig cfg);
    void run();

private:
    struct Sample {
        std::vector<float> feat, target, mask;   // target/mask [2*kCombos]
    };

    // env positioned at a random street root (random board, random pot via
    // a random legal prefix); returns false if the prefix ended the hand.
    bool sample_street_root(poker_ppo::PokerEnvironment& env,
                            std::mt19937& rng, int target_round);
    // Exact-solve the river PBS at the env's CURRENT state (must be a river
    // root) and append a training sample. beta is card-masked, unnormalized.
    bool river_sample_at(poker_ppo::PokerEnvironment& env, const uint8_t* b5,
                         const HunlPBS& beta, std::vector<Sample>& out);
    void direct_river_episode(poker_ppo::PokerEnvironment& env,
                              std::mt19937& rng, std::vector<Sample>& fresh);
    std::vector<double> random_range(std::mt19937& rng);
    // appends this episode's target (if any) to `fresh` (probed before
    // being merged into the replay)
    void self_play_episode(poker_ppo::PokerEnvironment& env, std::mt19937& rng,
                           std::vector<Sample>& fresh);
    double train_net();
    // Masked MSE of the net on the given samples. Called on each epoch's
    // FRESH samples BEFORE they are trained on — a true out-of-sample
    // generalization probe (in-replay MSE is memorization at small scale).
    double heldout_mse(const std::vector<Sample>& fresh);
    // Internal exploitability of a turn solve on a FIXED probe situation
    // with the given oracle — comparing net leaves vs exact leaves on the
    // same root is the cross-validation of the trained net.
    double probe_turn_expl(poker_ppo::PokerEnvironment& env,
                           HunlValueOracle* oracle, int T, int refresh_every);

    EndgameConfig cfg_;
    double        stack_;
    torch::Device device_;
    HunlValueNet  net_{nullptr};
    std::unique_ptr<torch::optim::Adam> opt_;
    std::mt19937  rng_;

    std::vector<Sample> replay_;
    long                seen_ = 0;
};

}  // namespace rebel_hunl
