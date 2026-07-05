// ReBeL (Brown, Bakhtin, Lerer, Gong 2020, arXiv:2007.13544) — faithful
// research implementation on the validated small games (Kuhn / N-card Kuhn /
// Leduc), judged by EXACT best response. Stage 1 of the HUNL port, following
// the repo's methodology: prove every theoretical component on games with an
// exact judge before any HUNL wiring.
//
// Faithful to the paper:
//   - PUBLIC BELIEF STATE (PBS): public betting history + revealed public
//     card + per-player beliefs over private DECK CARDS (not ranks — card
//     removal is exact with the product-form-plus-disjointness-mask
//     representation the paper uses for poker ranges).
//   - Depth-limited subgames: root PBS to the end of the current betting
//     round; leaves at the round boundary are CHANCE-resolved next-round
//     PBSs valued by the value net v̂(β). Final-round subgames extend to
//     terminals (exact range-vs-range payoffs) — the poker structure of the
//     paper (river subgames need no net).
//   - CFR-D with CFR-AVG leaf beliefs: each solver iteration recomputes leaf
//     PBS beliefs from the RUNNING AVERAGE strategy profile and re-queries
//     v̂ (paper §5). Solver is vectorized public-tree CFR+ (RM⁺, alternating
//     updates, linearly-weighted average).
//   - Per-infostate value targets: v_p(x|β) = root counterfactual value of
//     the final average profile divided by the opponent's compatible belief
//     mass — E[u_p | hold x, opp ~ β_opp\{x}, both play the solve's π̄].
//     Trained by masked MSE (mask: own belief > 0, x compatible).
//   - Algorithm 1's RANDOM-ITERATION continuation: t* ~ U{1..T}; at
//     iteration t* the next root is a LEAF PBS sampled by walking the
//     average policy at t* (CFR-AVG pairing), with ε-uniform exploration by
//     one randomly chosen player per episode. This matches the self-play
//     distribution of training PBSs to the distribution the solver queries —
//     the paper's key requirement for the fixed point to be a Nash.
//   - Value bootstrapping: round-2 subgames solve to terminal, so their root
//     targets ground the net; round-1 targets bootstrap through v̂ at the
//     round boundary.
//
// Documented deviations (choices the paper explicitly allows):
//   - No policy net: the paper uses it only to warm-start subgame CFR (§5,
//     "optional"); theory rests on the value net. Omitted for clarity.
//   - Test-time play re-solves each PBS with the trained v̂ and plays the
//     average policy (paper's default; no safe-resolving gadget). The
//     exploitability we report is the EXACT full-game best response against
//     the induced strategy, including subgames our average policy never
//     reaches (beliefs there fall back to uniform-over-compatible — flagged,
//     conservative).
#pragma once

#include <torch/torch.h>

#include <array>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "games.h"
#include "solvers.h"

namespace rebel {

// ─── deck-level view of an escher::Game ────────────────────────────────────
// Private hands are DECK-CARD indices; `rank[c]` maps to the rank the game's
// terminal utilities expect. Beliefs live over deck cards, so blocking
// (x ≠ y ≠ pub) is exact even with duplicated ranks (Leduc).
struct DeckSpec {
    std::vector<int> rank;      // deck card -> rank
    bool has_public = false;    // one public card revealed after round 1
    int n() const { return static_cast<int>(rank.size()); }
};

DeckSpec kuhn_deck(int n_cards);   // {0..n-1}, no public card
DeckSpec leduc_deck();             // {0,0,1,1,2,2}, public card

// ─── public belief state ───────────────────────────────────────────────────
struct PBS {
    std::string h;              // public betting history ("" = hand start)
    int pub = -1;               // revealed public DECK card, -1 before reveal
    std::vector<double> b0, b1; // beliefs over deck cards (sum 1; 0 at pub)

    const std::vector<double>& belief(int p) const { return p == 0 ? b0 : b1; }
    static PBS root(const DeckSpec& d);
};

// ─── leaf value oracle: v[p][x] = E[u_p | hold x, opp ~ β.b_{-p}\{x}] ──────
struct ValueOracle {
    virtual ~ValueOracle() = default;
    virtual void value(const PBS& beta,
                       std::array<std::vector<double>, 2>& out) = 0;
};

// ─── depth-limited public-tree CFR+ with decomposition ─────────────────────
class SubgameSolver {
public:
    struct Node {
        enum Kind { Decision, Terminal, ChanceLeaf };
        Kind kind;
        std::string h;
        int player = -1;                 // Decision
        std::vector<int> acts;           // legal actions (Decision)
        std::vector<int> child;          // per act index into nodes_
        // per-hand learning state, [card * n_acts + a] (Decision):
        std::vector<double> regret;      // RM⁺ cumulative (clamped ≥ 0)
        std::vector<double> cum_strat;   // linearly-weighted average policy
    };

    // cfr_avg: leaf PBS beliefs from the running AVERAGE profile (paper's
    // CFR-AVG) vs the CURRENT iterate (CFR-D — the variant Theorem 3 is
    // stated for; regret updates are exactly consistent with these beliefs).
    SubgameSolver(const escher::Game& g, const DeckSpec& deck, PBS root,
                  ValueOracle* oracle, bool cfr_avg = true);

    // One CFR+ iteration (leaf-belief refresh + oracle queries + alternating
    // regret updates for both players). `t` is 1-based (linear averaging).
    void iterate(int t);

    // Root per-infostate values of the FINAL average profile (normalized by
    // opponent compatible mass). mask[p][x] = target well-defined.
    void root_values(std::array<std::vector<double>, 2>& v,
                     std::array<std::vector<double>, 2>& mask);

    // Algorithm 1 SampleLeaf under the CURRENT average policy (CFR-AVG
    // pairing): walks sampled private hands from the root, ε-uniform
    // exploration for `explorer`'s decisions, chance uniform over compatible
    // cards. Returns false if the walk ended at a subgame-internal terminal
    // (episode over); else fills `out` with the leaf PBS (next-round root).
    bool sample_leaf(std::mt19937& rng, double eps, int explorer, PBS* out);

    // Average policy over legal actions at a decision node for `card`
    // (uniform where the card never accumulated weight).
    std::vector<double> avg_policy(int node, int card) const;

    const std::vector<Node>& nodes() const { return nodes_; }
    const PBS& root() const { return root_; }

    // Reach vectors at node `i` for both players under the average (or, with
    // avg=false, the current RM⁺) profile (belief ⊙ own-strategy products;
    // used for leaf beliefs, SampleLeaf, evaluation).
    void avg_reaches(int i, std::vector<double>& r0, std::vector<double>& r1,
                     bool avg = true) const;

private:
    int build(const std::string& h);
    void refresh_leaves(bool avg);              // leaf belief re-queries
    // Alternating-update walk for `upd`: returns counterfactual values for
    // upd per own card (opponent reach, pair mask and chance folded in).
    std::vector<double> walk(int node, int upd, int t,
                             std::vector<double>& my_reach,
                             std::vector<double>& opp_reach);
    std::vector<double> current_policy(const Node& nd, int card) const;
    const std::vector<double>& terminal_u0(int node);  // [x*n+y] P0 utility

    const escher::Game& g_;
    const DeckSpec&     deck_;
    PBS                 root_;
    ValueOracle*        oracle_;
    bool                cfr_avg_;
    int                 n_;
    std::vector<Node>   nodes_;
    std::unordered_map<int, std::vector<double>> term_u0_;      // node -> nxn
    // chance-leaf cache, refreshed each iteration: values_[node][pub][p][x]
    std::unordered_map<int, std::vector<std::array<std::vector<double>, 2>>>
        leaf_v_;
};

// ─── value net over PBS features ───────────────────────────────────────────
// Input: [round2-root-history one-hot | pub one-hot(n) | b0(n) | b1(n)].
// Output: 2n — v0[x], v1[x] in each player's own frame.
struct ValueNetImpl : torch::nn::Module {
    torch::nn::Linear l1{nullptr}, l2{nullptr}, l3{nullptr};
    ValueNetImpl(int d_in, int d_out, int hidden);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ValueNet);

class Featurizer {
public:
    Featurizer(const escher::Game& g, const DeckSpec& deck);
    std::vector<float> operator()(const PBS& beta) const;
    int dim() const { return dim_; }
    int n_hist() const { return static_cast<int>(hidx_.size()); }

private:
    std::unordered_map<std::string, int> hidx_;  // round-2 root histories
    int n_, dim_;
};

// Exact recursive oracle (validation): solves the next-round subgame to
// terminal with `iters` CFR+ iterations and returns its root values. With
// this oracle the round-1 solve is exact CFR-D — the decomposition test.
class ExactOracle : public ValueOracle {
public:
    ExactOracle(const escher::Game& g, const DeckSpec& d, int iters)
        : g_(g), deck_(d), iters_(iters) {}
    void value(const PBS& beta,
               std::array<std::vector<double>, 2>& out) override;

private:
    const escher::Game& g_;
    const DeckSpec&     deck_;
    int                 iters_;
};

class NetOracle : public ValueOracle {
public:
    NetOracle(ValueNet net, const Featurizer& f) : net_(net), feat_(f) {}
    void value(const PBS& beta,
               std::array<std::vector<double>, 2>& out) override;

private:
    ValueNet          net_;
    const Featurizer& feat_;
};

// ─── ReBeL training loop (Algorithm 1) ─────────────────────────────────────
struct Config {
    int    t_train      = 250;    // CFR iterations per self-play solve
    int    t_eval       = 1000;   // iterations for evaluation-time solves
    int    episodes     = 128;    // self-play episodes per epoch
    int    epochs       = 40;
    int    sgd_steps    = 500;
    int    batch        = 512;
    int    hidden       = 256;
    double lr           = 1e-3;
    double eps_explore  = 0.25;   // Algorithm 1 exploration
    int    replay_cap   = 200000;
    int    exact_iters  = 400;    // ExactOracle inner solve (validation)
    bool   cfr_avg      = true;   // leaf beliefs: average (paper) vs current
    // Evaluation-only: mix `smooth`·uniform into chance-leaf beliefs when
    // building the induced strategy — diagnoses how much measured
    // exploitability comes from UNREACHED subgames (zero-mass beliefs →
    // uniform fallback → free exploits for the BR; the unsafe-resolving gap).
    double eval_smooth  = 0.0;
    uint64_t seed       = 0;
};

class Trainer {
public:
    Trainer(const escher::Game& g, DeckSpec deck, Config cfg);
    void run();

    // Induced full-game strategy of the ReBeL agent (re-solve every PBS the
    // public tree can reach, beliefs propagated by the solves' averages).
    escher::Strategy agent_strategy(ValueOracle* oracle, int t_solve);

private:
    void self_play_episode(std::mt19937& rng);
    void push_targets(SubgameSolver& s);
    double train_value_net();
    double value_probe_mse(int k);  // net vs ExactOracle on replay samples

    const escher::Game& g_;
    DeckSpec            deck_;
    Config              cfg_;
    Featurizer          feat_;
    ValueNet            net_{nullptr};
    std::unique_ptr<torch::optim::Adam> opt_;
    std::mt19937        rng_;

    struct Sample {
        std::vector<float> feat;
        std::vector<float> target, mask;   // [2n]
        PBS beta;                          // kept for the probe diagnostic
    };
    std::vector<Sample> replay_;
    long                replay_seen_ = 0;
};

}  // namespace rebel
