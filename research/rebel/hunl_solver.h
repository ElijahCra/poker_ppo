// HUNL street-limited public subgame solver (ReBeL stage 2).
//
// The betting tree is built by REPLAYING a PokerEnvironment with
// push_state/pop_state, so pot arithmetic, raise sizing, all-in handling and
// legality are byte-identical to the abstraction the trainers and LBR use —
// no separate rules re-implementation to drift.
//
// Solver: per-combo (1326) vectorized public-tree CFR+ (RM⁺, alternating
// updates, linear averaging) — the validated Leduc structure scaled up.
// Terminals inside the street:
//   Fold          — inclusion-exclusion fold kernel
//   Showdown      — complete board: exact rank-sweep kernel
//   AllinShowdown — all-in call one card early: exact enumeration of the
//                   remaining card through the showdown kernel (uniform
//                   chance, per-combo card removal). Multi-street all-ins
//                   (flop/preflop subgames) are a later stage and guarded.
// StreetEnd nodes (betting closed, next street pending) are the depth-limit
// leaves: for each candidate next card, a ValueOracle values the
// chance-resolved next-street PBS at CFR-AVG beliefs — exact next-street
// re-solve for validation, the value net in training.
//
// `allowed_actions` optionally restricts the abstraction (e.g. {fold, call,
// pot, all-in}) — subgame solvers routinely use sparser action sets than the
// blueprint; internal BR is measured in the same restricted game, so the
// validation ladder stays exact.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "hunl_kernels.h"
#include "poker_env.h"

namespace rebel_hunl {

struct HunlPBS {
    // Public state is carried by the ENV (board + pot + betting position);
    // the solver only needs the ranges.
    std::vector<double> r0, r1;   // combo weights (normalized over valid)
};

struct HunlValueOracle {
    // env sits at the next-street root reached by path replay (its randomly
    // dealt card is irrelevant — the candidate board is passed explicitly).
    // out[p][combo] = E[chips_p | hold combo, opp ~ beta.r_{-p} \ overlap].
    virtual ~HunlValueOracle() = default;
    virtual void value(poker_ppo::PokerEnvironment& env, const uint8_t* board,
                       int nb, const HunlPBS& beta,
                       std::array<std::vector<double>, 2>& out) = 0;
    // All candidate next cards of one leaf in a single call: base board is
    // the street's nb_base cards; row i uses base+cards[i]. Default loops
    // value(); net oracles override with ONE batched forward — the entire
    // GPU win for leaf queries.
    virtual void value_batch(poker_ppo::PokerEnvironment& env,
                             const uint8_t* base_board, int nb_base,
                             const std::vector<uint8_t>& cards,
                             const std::vector<HunlPBS>& betas,
                             std::vector<std::array<std::vector<double>, 2>>&
                                 outs) {
        outs.resize(cards.size());
        std::array<uint8_t, 5> b{};
        for (int i = 0; i < nb_base; ++i) b[i] = base_board[i];
        for (size_t i = 0; i < cards.size(); ++i) {
            b[nb_base] = cards[i];
            value(env, b.data(), nb_base + 1, betas[i], outs[i]);
        }
    }
};

class HunlSolver {
public:
    struct Node {
        enum Kind { Decision, Fold, Showdown, AllinShowdown, StreetEnd };
        Kind kind;
        int player = -1;              // Decision: actor; Fold: the folder
        std::vector<int> acts;        // env action ids
        std::vector<int> child;
        std::array<int, 2> contrib{}; // chips invested up to this node
        std::vector<int> path;        // env actions root -> node
        std::vector<double> regret, cum_strat;   // [combo * |acts| + k]
    };

    // Builds the tree from the env's CURRENT state (restored on return).
    // board_override (nb cards) replaces the env's board for all card logic —
    // used by oracles whose env carries an arbitrary dealt card.
    HunlSolver(poker_ppo::PokerEnvironment& env, HunlPBS root,
               HunlValueOracle* oracle,
               std::vector<int> allowed_actions = {},
               const uint8_t* board_override = nullptr, int nb_override = 0);

    void iterate(int t);
    // Refresh StreetEnd leaf values every k iterations instead of every one.
    // Leaf beliefs move slowly under averaging, so k>1 is a mild, documented
    // approximation — intended for the EXACT-oracle validation mode, where
    // each refresh costs |leaves|×48 full next-street re-solves. Net oracles
    // are cheap enough to keep k=1.
    int refresh_every = 1;
    // Subgame exploitability of the current average profile in chips/hand
    // (normalized by joint compatible mass; 0 at equilibrium). Exact when
    // there are no StreetEnd leaves; with an oracle it is BR within the
    // depth-limited game (leaf values held fixed at final-average beliefs).
    double exploitability();
    // TRUE composed two-street exploitability: re-solves every (leaf, card)
    // river subgame at the final-average leaf beliefs (the agent's
    // deployment rule), then best-responds through BOTH streets — unlike
    // exploitability(), whose BR is confined to the leaf-valued game. The
    // Leduc-grade full-game judge for turn+river endgames.
    // Costs |leaves| × ~48 river solves of t_river iterations.
    double exploitability_composed(int t_river);
    // BR values for player p vs the current average profile with an explicit
    // (unnormalized) opponent reach vector — counterfactual sums, the
    // building block exploitability_composed uses across streets.
    std::vector<double> best_response(int p, std::vector<double> opp_reach);
    // Root per-combo values of the average profile (normalized by opponent
    // compatible mass); mask = target well-defined.
    void root_values(std::array<std::vector<double>, 2>& v,
                     std::array<std::vector<double>, 2>& mask);

    // ── Iterate-averaged root values (ReBeL's value target) ────────────
    // The paper trains v̂ on (1/T)·Σ_t v^{π^t}(β_r) — the running average
    // of CURRENT-iterate root values, not the final average profile's
    // value. Call track_root_values() before the first iterate();
    // avg_root_values() returns per-combo values in the root_values
    // normalization, or false if nothing was tracked.
    void track_root_values() { track_root_ = true; }
    bool avg_root_values(std::array<std::vector<double>, 2>& v,
                         std::array<std::vector<double>, 2>& mask) const;

    std::vector<double> avg_policy(int node, int combo) const;

    // ── Safe re-solving gadget (Burch et al. 2014) ─────────────────────
    // Replaces player `opp`'s fixed root range with a per-combo
    // Terminate/Follow decision, regret-matched across iterations:
    // Terminate takes alt[combo] (that holding's per-combo value estimate
    // from the PRIOR solve — normalized convention, chips); Follow enters
    // the subgame. The re-solved average strategy then concedes no
    // holding more than its alternative — safety against opponents who
    // deviate from the tracked range (measured: LBR farmed exactly that
    // on the river). The tracked range remains only the entry PRIOR,
    // floored by `mix` uniform over valid combos so EVERY holding keeps
    // entry mass. Call after construction, before the first iterate().
    // Intended for leafless (river) solves; root_values/exploitability
    // afterwards read the entry prior, not the tracked range.
    void enable_gadget(int opp, const std::vector<double>& alt,
                       double mix = 0.1);

    // Per-combo values of BOTH players at an interior node under the
    // final average profile (gadget entry averaged in), normalized by the
    // opposing compatible reach mass — the alt-value source when
    // re-solving further down the line of play. mask = well-defined.
    void values_at(int node, std::array<std::vector<double>, 2>& v,
                   std::array<std::vector<double>, 2>& mask);

    // Average-profile reaches to a node (gadget entry averaged in) — the
    // CFR-AVG beliefs a deployed agent carries across a street boundary.
    void beliefs_at(int node, std::vector<double>& r0,
                    std::vector<double>& r1) const;

    const std::vector<Node>& nodes() const { return nodes_; }
    const std::vector<uint8_t>& valid() const { return valid_; }
    int board_count() const { return nb_root_; }
    const std::array<uint8_t, 5>& board() const { return board_; }

    // Algorithm 1 SampleLeaf under the current AVERAGE profile (CFR-AVG
    // pairing): samples compatible private hands from the root ranges, walks
    // sampled actions (ε-uniform for `explorer`'s decisions), samples the
    // next card at a StreetEnd. Returns false if the walk hit a terminal
    // (episode over); else fills leaf node id, card, and the leaf PBS
    // (avg-policy Bayes posteriors, card-masked, normalized).
    bool sample_leaf(std::mt19937& rng, double eps, int explorer,
                     int* leaf_node, uint8_t* card, HunlPBS* beta);

private:
    int build(std::vector<int> path);
    void refresh_leaves();
    // Alternating-update / value walk. update=false → both play the average
    // profile, no state changes (used for root values and BR terminals).
    std::vector<double> walk(int i, int upd, int t, bool update,
                             std::vector<double>& my_reach,
                             std::vector<double>& opp_reach);
    std::vector<double> br_walk(int i, int p, std::vector<double>& opp_reach);
    std::vector<double> policy_row(const Node& nd, int combo,
                                   bool average) const;
    // flat [combo*A+k] policies for a whole node in one pass (no per-combo
    // allocations — the walk's hot path)
    void policies_into(const Node& nd, bool average,
                       std::vector<double>& out) const;
    void reaches_to(int target, bool average, std::vector<double>& r0,
                    std::vector<double>& r1) const;

    poker_ppo::PokerEnvironment& env_;
    HunlPBS              root_;
    HunlValueOracle*     oracle_;
    std::vector<int>     allowed_;
    // gadget state: per-combo T/F regrets (RM⁺), current & cumulative
    // Follow probability (linear-weighted like cum_strat), and the
    // mass-weighted alternative payoffs (alt · our compatible reach)
    int                  gadget_opp_ = -1;
    std::vector<double>  gd_alt_w_, gd_rT_, gd_rF_, gd_pF_, gd_cumF_;
    double               gd_cumW_ = 0.0;
    bool                 track_root_ = false;
    std::array<std::vector<double>, 2> rv_sum_;
    long                 rv_n_ = 0;
    int                  root_round_ = 0;
    int                  nb_root_ = 0;
    std::array<uint8_t, 5> board_{};
    std::vector<uint8_t> valid_;
    std::vector<Node>    nodes_;
    std::vector<int>     leaf_ids_;
    // StreetEnd leaf values per node per candidate card: [node][card][p][combo]
    std::vector<std::vector<std::array<std::vector<double>, 2>>> leaf_v_;
    // board-cached showdown evaluators: [0] = root board (river subgames);
    // per-runout entries for AllinShowdown (keyed by the runout card)
    std::unique_ptr<RiverEval> river_eval_;
    std::array<std::unique_ptr<RiverEval>, kCards> runout_eval_;
    const RiverEval& eval_for_root();
    const RiverEval& eval_for_runout(uint8_t c);
};

// Exact next-street oracle for validation: re-solves the next street with
// `iters` CFR+ iterations under the same action restriction.
class ExactStreetOracle : public HunlValueOracle {
public:
    ExactStreetOracle(int iters, std::vector<int> allowed = {})
        : iters_(iters), allowed_(std::move(allowed)) {}
    void value(poker_ppo::PokerEnvironment& env, const uint8_t* board, int nb,
               const HunlPBS& beta,
               std::array<std::vector<double>, 2>& out) override;

private:
    int iters_;
    std::vector<int> allowed_;
};

}  // namespace rebel_hunl
