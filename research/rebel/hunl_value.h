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
    // `layers` = hidden layers. layers=2 registers exactly {l1,l2,l3} —
    // checkpoint-compatible with every net trained before the knob existed.
    // gelu_ln=true switches hidden activations to LayerNorm+GeLU — the
    // ReBeL paper's spec (6×1536, GeLU, LayerNorm). Default stays ReLU so
    // existing checkpoints keep loading (LayerNorm adds parameters).
    torch::nn::Linear l1{nullptr}, l3{nullptr};
    std::vector<torch::nn::Linear> mids;
    std::vector<torch::nn::LayerNorm> lns;
    bool gelu = false;
    HunlValueNetImpl(int hidden, int layers = 2, bool gelu_ln = false);
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
    int    layers       = 2;     // hidden layers (2 = the original l1/l2/l3)
    bool   gelu_ln      = false; // LayerNorm+GeLU hiddens (ReBeL spec)
    double lr           = 1e-3;
    double lr_final     = 0.0;   // >0: linear lr decay to this over epochs
    // pointwise Huber on pot-unit errors (both papers use Huber): quadratic
    // below delta, linear above — caps the gradient of the rare huge-error
    // rows (small-pot all-in situations where |target| can reach stack/pot).
    // <=0 falls back to plain MSE (A/B escape hatch; note vloss scale:
    // Huber = MSE/2 in the quadratic regime).
    double huber_delta  = 1.0;
    double eps_explore  = 0.25;
    int    replay_cap   = 300000;   // ~32KB/sample → ~10GB resident
    // turn probes are single-situation BR bounds — noisy. probe_k averages
    // over K fixed situations (K exact references solved once at startup);
    // 0 skips probes entirely (offline screens that only need heldout).
    int    probe_k      = 1;
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
    // >0: solve river targets in GPU lockstep batches of this size
    // (BatchRiverSolver; equivalence-validated vs the CPU solver). CPU
    // workers build specs; the device does the solving.
    int    gpu_batch    = 0;
    // net checkpoint: loaded at start if present, saved each epoch. Lets a
    // train_turn run fine-tune a river-pretrained net (and later serves the
    // play-time solver). Empty = off.
    std::string ckpt    = "rebel_value.pt";
    // Sample persistence. Every solver target costs ~0.2 core-seconds of
    // exact CFR; without a dataset file the whole stream dies with the
    // process and every architecture experiment re-pays the solves.
    //   data_out: append every fresh sample (net-independent exact targets)
    //   data_in:  load at startup — a fixed 2% slice (row % 50, cap 8192)
    //             becomes a persistent heldout set (comparable across
    //             configs), the rest reservoir-fills the replay.
    // data_in + episodes=0 = offline training: no sampling, epochs are
    // pure SGD on the loaded replay, judged on the fixed heldout split.
    std::string data_in, data_out;
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
    // GPU path: CPU workers sample specs (board/pot/ranges/tree shape), the
    // device solves them in lockstep batches; appends `episodes` samples.
    void gpu_river_epoch(
        std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>>& envs,
        int W, int ep, std::vector<Sample>& fresh);
    // Random training range for the situation's board (nb=5 river, nb=4
    // turn). 70% DeepStack R(S,p): recursive mass splits over the valid
    // combos ORDERED BY HAND STRENGTH (weaker half / stronger half) —
    // CFR ranges are strength-polarized, and a net trained on shuffled
    // partitions never sees that structure. 30% iid coverage floor.
    std::vector<double> random_range(std::mt19937& rng, const uint8_t* board,
                                     int nb);
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
    // (one per seed) with the given oracle — comparing net leaves vs exact
    // leaves on the same root is the cross-validation of the trained net.
    // Builds its own env from the seed: a shared env's deal RNG advances
    // across resets, which silently made every probe a DIFFERENT board
    // (the old log's probe scatter was situation variance, not just net
    // drift, and the exact reference was a different situation entirely).
    double probe_turn_expl(HunlValueOracle* oracle, int T, int refresh_every,
                           uint32_t situation_seed);
    // dataset file: [magic|version|kdim|nout] header, then fixed-size rows
    // of feat f32[kdim] | target f32[nout] | mask u8[nout]. feat embeds the
    // raw situation (street/board one-hots, pot/stack, both normalized
    // ranges) so a future featurizer can re-derive its inputs from it.
    void load_dataset(const std::string& path);
    void append_dataset(const std::string& path,
                        const std::vector<Sample>& fresh);
    // reservoir-insert into replay_ (uniform over all samples ever seen)
    void replay_insert(Sample&& smp);

    EndgameConfig cfg_;
    double        stack_;
    torch::Device device_;
    HunlValueNet  net_{nullptr};
    std::unique_ptr<torch::optim::Adam> opt_;
    std::mt19937  rng_;

    std::vector<Sample> replay_;
    std::vector<Sample> heldout_;   // fixed split from data_in (never trained)
    long                seen_ = 0;
};

}  // namespace rebel_hunl
