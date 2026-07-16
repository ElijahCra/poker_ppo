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
#include <unordered_map>
#include <vector>

#include "hunl_solver.h"

namespace rebel_hunl {

class HunlFeaturizer {
public:
    // [street(4) | board(52) | pot/stack | r0 | r1 | pct | eq0 | eq1]
    // pct/eq blocks are the hand-strength features (2026-07-07): raw card
    // one-hots force the net to learn hand evaluation implicitly — the
    // measured error floor after capacity (screen) and distribution
    // (train_turn) were ruled out. pct = board-strength percentile
    // (range-independent anchor); eq_p = per-combo equity of player p's
    // hands against the OPPONENT'S RANGE (win−lose mass / compatible mass,
    // exact card removal ∈ [−1,1]) — the board×range interaction, computed
    // by the same kernels the exact solver uses. Ranges stay at the same
    // offsets: the zero-sum layer's slices are unchanged.
    static constexpr int kDim = 4 + kCards + 1 + 5 * kCombos;
    // contribs equal at a street root; pot = 2*contrib.
    // Computes strength features internally (exact for nb==5; zero for
    // nb<5 — the net is only ever queried at river roots today; Phase C's
    // turn-root queries need the E-over-river-card version).
    static std::vector<float> features(int street, const uint8_t* board,
                                       int nb, double pot, double stack,
                                       const HunlPBS& beta);
    // precomputed-strength variant (solver leaf queries: the per-card
    // rank/sort context is cached across refreshes, see HunlNetOracle)
    static std::vector<float> features(int street, const uint8_t* board,
                                       int nb, double pot, double stack,
                                       const HunlPBS& beta,
                                       const std::vector<double>& pct,
                                       const std::vector<double>& eq0,
                                       const std::vector<double>& eq1);
};

struct HunlValueNetImpl : torch::nn::Module {
    // `layers` = hidden layers. layers=2 registers exactly {l1,l2,l3} —
    // checkpoint-compatible with every net trained before the knob existed.
    // gelu_ln=true switches hidden activations to LayerNorm+GeLU — the
    // ReBeL paper's spec (6×1536, GeLU, LayerNorm). Default stays ReLU so
    // existing checkpoints keep loading (LayerNorm adds parameters).
    // zero_sum=true appends DeepStack's outer zero-sum step (differentiable,
    // so training sees it, exactly as the paper trains through it). In OUR
    // value convention (per-infostate values normalized by opponent
    // compatible mass) the game-value identity is
    //   Σ_i r0_i·m0_i·v0_i + Σ_j r1_j·m1_j·v1_j = 0,
    // m = opponent mass compatible with the combo (card removal); the
    // violation, split evenly, is subtracted from every value entry.
    torch::nn::Linear l1{nullptr}, l3{nullptr};
    std::vector<torch::nn::Linear> mids;
    std::vector<torch::nn::LayerNorm> lns;
    bool gelu = false;
    bool zero_sum = true;
    // [kCombos, kCards] combo→card membership. Deliberately NOT a
    // registered buffer: it's a deterministic constant, and serializing it
    // would break loading pre-zero-sum checkpoints.
    torch::Tensor zs_M;
    HunlValueNetImpl(int hidden, int layers = 2, bool gelu_ln = false,
                     bool zero_sum_on = true);
    void to(torch::Device device, bool non_blocking = false) override;
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(HunlValueNet);

class HunlNetOracle : public HunlValueOracle {
public:
    HunlNetOracle(HunlValueNet net, double stack, torch::Device device)
        : net_(net), stack_(stack), device_(device) {}
    // per-candidate-card strength context, cached across leaf refreshes
    // (the ~48 candidate boards repeat every refresh; ranking+sorting per
    // query would dominate solve time). Keyed by card; guarded by a
    // base-board signature. One oracle per solve/episode — no locking.
    // River queries (base=4 cards) use `ev`; turn-root queries (base=3,
    // flop solves) use the per-runout array + precomputed E[percentile].
    struct CardCtx {
        std::unique_ptr<RiverEval> ev;
        std::vector<double> pct;
        std::array<std::unique_ptr<RiverEval>, kCards> runout;
    };
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
    // pot-parameterized core — the env above is consulted ONLY for the
    // pot. Callers that know the leaf pot (the batched turn pipeline
    // reads it off spec contribs) skip env plumbing entirely.
    void value_batch_pot(
        double pot, const uint8_t* base_board, int nb_base,
        const std::vector<uint8_t>& cards, const std::vector<HunlPBS>& betas,
        std::vector<std::array<std::vector<double>, 2>>& outs);
    // Split API for callers that batch MANY leaves into ONE forward
    // (2400 small per-leaf forwards measured as 98% of the batched turn
    // solver's wall): river_row_features writes one river-root query row
    // (cached per-card strength context; NOT thread-safe on one oracle —
    // distinct oracles are independent); values_forward runs the net on
    // a [N, kDim] CPU feature block and returns [N, 2*kCombos] on CPU,
    // in POT units (callers scale by their pots).
    void river_row_features(const uint8_t* board4, uint8_t card, double pot,
                            const HunlPBS& beta, float* dst);
    torch::Tensor values_forward(const torch::Tensor& feats_cpu);
    // preflop leaves: one forward for a whole leaf's sampled flops (the
    // fixed set recurs every leaf × refresh AND across solves — pf_seed
    // is fixed per agent — so the per-flop runout evaluators live in
    // flop_ctx_, not in HunlFeaturizer)
    void value_boards(poker_ppo::PokerEnvironment& env,
                      const std::vector<std::array<uint8_t, 3>>& flops,
                      const std::vector<HunlPBS>& betas,
                      std::vector<std::array<std::vector<double>, 2>>& outs)
        override;
    // per-flop strength context: the 32 deterministic runout evaluators +
    // E[percentile] are beta-independent (cached); equity blocks depend
    // on beta and are recomputed per query from the cached evaluators.
    // MUST replicate HunlFeaturizer's nb==3 math bit-exactly — training
    // rows use the uncached path; `rebel_hunl flopctx_check` verifies.
    struct FlopCtx {
        std::array<std::array<uint8_t, 2>, 32> runouts{};
        std::array<std::unique_ptr<RiverEval>, 32> ev;
        std::vector<double> pct, nrun;
    };
    // cache-backed replica of HunlFeaturizer::features(1, flop, 3, ...) —
    // public so flopctx_check can compare the two paths
    std::vector<float> flop_features(const uint8_t* flop, double pot,
                                     const HunlPBS& beta);

private:
    HunlValueNet net_;
    double stack_;
    torch::Device device_;
    std::array<CardCtx, kCards> ctx_{};
    std::array<uint8_t, 5> ctx_board_{255, 255, 255, 255, 255};
    int ctx_nb_ = -1;
    // keyed by sorted 3-card triple (matches the featurizer's sorted
    // runout hash: any ordering of a flop shares one context)
    std::unordered_map<uint32_t, FlopCtx> flop_ctx_;
};

// Migrate a dataset file written by the pre-hand-strength featurizer
// (kdim 2709) to the current layout: decode the raw situation from the
// stored feat (board multi-hot, pot/stack, both ranges), recompute the
// strength blocks, keep target/mask byte-identical. The exact solver
// targets cost ~0.2 core-seconds each — never regenerate what a
// featurizer change can convert.
int convert_dataset(const std::string& in, const std::string& out);

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
    bool   zero_sum     = true;  // DeepStack outer zero-sum correction
    // Replay policy. ReBeL uses a CIRCULAR buffer ("a simple circular
    // buffer of size 12M and sample uniformly") — a recency window, correct
    // when the sample distribution tracks the improving net (train_turn).
    // For stationary river targets, reservoir (uniform over all history)
    // keeps more diversity. -1 = auto: circular for train_turn, reservoir
    // for train_river.
    int    circular     = -1;
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
    // train_flop mode: direct flop-situation sampling — flop subgames
    // solved with net TURN-root leaves; emits one street-1 (flop-root)
    // iterate-avg row per solve PLUS `harvest` street-2 rows from
    // net-leaf turn solves at the flop solve's own (leaf, card) query
    // distribution — the street-2 volume machine the flop lever needs.
    bool   flop_mode    = false;
    int    t_flop       = 60;
    // train_root mode: Algorithm 1 from the GAME ROOT — preflop solve at
    // uniform ranges (sampled-flop leaves), then flop/turn continuations
    // from its own sampled + harvested leaf PBSs. Street-1/2 rows land on
    // the DEPLOYED agent's query distribution instead of random_range
    // synthetics — the self-consistency closure. No street-0 rows are
    // emitted (no solver queries preflop-root values).
    bool   root_mode    = false;
    int    t_preflop    = 40;
    int    pf_samples   = 64;   // sampled flop leaves per preflop solve
    // PCFR+ for the CPU solves that produce training targets (turn/flop/
    // preflop subgame solves AND the exact river target solves). The GPU
    // river pipeline (gpu_batch) stays plain CFR+ — don't mix variants in
    // one dataset build if rows must be provenance-uniform.
    bool   pcfr         = false;
    // >0: solve river targets in GPU lockstep batches of this size
    // (BatchRiverSolver; equivalence-validated vs the CPU solver). CPU
    // workers build specs; the device does the solving.
    int    gpu_batch    = 0;
    // >0 (train_turn only): a GPU lane solves TURN targets in lockstep
    // batches of this size (BatchTurnSolver: fused kernel + on-device
    // leaf featurization, PCFR+-aware). A GPU episode yields ONE
    // street-2 row and NO river harvest — river rows come from the
    // river lanes — so size `episodes` accordingly. REBEL_GPU_FRAC
    // splits episodes between this lane (default 0.9) and CPU workers.
    int    gpu_turn_batch = 0;
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
    // train_flop episode: flop solve (net turn leaves) → street-1 root
    // row + street-2 rows from harvested net-leaf turn solves
    void flop_episode(poker_ppo::PokerEnvironment& env, std::mt19937& rng,
                      std::vector<Sample>& fresh);
    // solve the flop PBS at the env's CURRENT state (board_override b3:
    // the env may have dealt different cards) → street-1 row + street-2
    // turn continuations. Shared by flop_episode (random roots) and
    // root_episode (preflop-solve leaf PBSs).
    void flop_continue(poker_ppo::PokerEnvironment& env, const uint8_t* b3,
                       const HunlPBS& beta, std::mt19937& rng,
                       std::vector<Sample>& fresh);
    // train_root episode: Algorithm 1 from the TRUE game root — preflop
    // solve at uniform ranges, flop continuations at its own leaf PBSs
    void root_episode(poker_ppo::PokerEnvironment& env, std::mt19937& rng,
                      std::vector<Sample>& fresh);
    // emit the solver's iterate-averaged ROOT values as a training row
    // (shared by the turn/flop episode paths). beta = raw root ranges.
    void emit_root_sample(poker_ppo::PokerEnvironment& env, HunlSolver& s,
                          const HunlPBS& beta, const uint8_t* board, int nb,
                          int street, std::vector<Sample>& fresh);
    // GPU path: CPU workers sample specs (board/pot/ranges/tree shape), the
    // device solves them in lockstep batches; appends `episodes` samples.
    // Runs on its own env pool so it can PIPELINE with the CPU worker pool
    // (separate resources; sequential use wastes one of them).
    void gpu_river_epoch(
        std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>>& envs,
        int W, int ep, std::vector<Sample>& fresh, int episodes);
    // GPU turn lane (train_turn): CPU workers build TurnSpecs, the
    // device solves signature-grouped lockstep batches, street-2 rows
    // are emitted from iterate-averaged root values.
    void gpu_turn_epoch(
        std::vector<std::unique_ptr<poker_ppo::PokerEnvironment>>& envs,
        int W, int ep, std::vector<Sample>& fresh, int episodes);
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
    // insert into replay_: circular (overwrite oldest) or reservoir
    // (uniform over all samples ever seen), per cfg_.circular
    void replay_insert(Sample&& smp);
    // pot/stack coverage of a sample batch — DeepStack sampled pots
    // heavy-tailed on purpose; ours are prefix-induced, so verify the
    // big-pot tail isn't starved
    void print_pot_hist(const std::vector<Sample>& v, const char* tag);

    EndgameConfig cfg_;
    double        stack_;
    torch::Device device_;
    HunlValueNet  net_{nullptr};
    std::unique_ptr<torch::optim::Adam> opt_;
    std::mt19937  rng_;

    std::vector<Sample> replay_;
    std::vector<Sample> heldout_;   // fixed split from data_in (never trained)
    long                seen_ = 0;
    bool                circular_ = false;   // resolved from cfg_.circular
};

}  // namespace rebel_hunl
