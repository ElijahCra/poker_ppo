// Batched TURN subgame solver on torch tensors (GPU path, stage 2).
//
// Solves B turn subgames in LOCKSTEP (one shared betting-tree topology,
// per-spec boards / pots / ranges), CFR+ or PCFR+ — the same math as the
// CPU HunlSolver on a turn root, judged by `turn_gpu_check` equivalence.
//
// Leaf semantics (matching HunlSolver::walk exactly):
//   Fold           — compatible-mass matmul, same as the river solver.
//   AllinShowdown  — 1 card to come: average over the 48 runout rivers of
//                    the sorted-prefix showdown kernel, each with the
//                    runout card masked from both sides, ÷ 44 (the exact
//                    per-pair runout count). Runouts ride the batch dim:
//                    [B·52, n] — 6 kernel launches for all runouts.
//   StreetEnd      — river-root leaves valued by the NET. The solver holds
//                    a per-leaf [B, 52, 2, n] value tensor (card slot ==
//                    runout river card; board-card slots dead) that the
//                    CALLER refreshes: download average-profile leaf
//                    beliefs (leaf_reaches), query the oracle, upload
//                    (set_leaf_values). Between refreshes the values are
//                    constants and iterations are pure tree algebra.
//                    cfv[x] += v_c[x] · (S − Sc[a] − Sc[b] + w[x]) / 44.
//
// The solver is torch-pure: no net, no env — the oracle round-trip lives
// in the caller (same division of labor as gpu_river_epoch's spec
// builders), which also guarantees feature/net parity with the CPU
// solver: both paths go through HunlNetOracle::value_batch.
#pragma once

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include "hunl_gpu.h"
#include "hunl_solver.h"
#include "hunl_value.h"

namespace rebel_hunl {

struct TurnSpec {
    std::array<uint8_t, 4> board{};
    std::vector<double> r0, r1;          // root ranges (unnormalized ok)
    std::vector<int> node_contrib0;      // per tree node, this spec's chips
    std::vector<int> node_contrib1;
};

class BatchTurnSolver {
public:
    BatchTurnSolver(TreeShape shape, std::vector<TurnSpec> specs,
                    torch::Device device, torch::Dtype dtype = torch::kFloat,
                    bool pcfr = false, bool pcfr_quad = true);
    ~BatchTurnSolver();   // out-of-line: CUDAGraph is fwd-declared here

    // StreetEnd leaves, in shape order. leaf_reaches/set_leaf_values are
    // indexed by position in this list.
    const std::vector<int>& leaf_nodes() const { return leaf_nodes_; }
    // env actions root → leaf j (to position an env for oracle pot/board)
    const std::vector<int>& leaf_path(int j) const {
        return leaf_path_[static_cast<size_t>(j)];
    }

    // Average-profile reaches to leaf j (root ranges INCLUDED, computed on
    // device, downloaded as doubles): out[b][player][combo]. The caller
    // masks per candidate card and queries the oracle — identical inputs
    // to the CPU solver's refresh_leaves.
    void leaf_reaches(
        int leaf_idx,
        std::vector<std::array<std::vector<double>, 2>>& out);
    // All leaves at once: ONE download of the average policies, path
    // walks on CPU threads. The per-leaf device walk costs ~30 kernel
    // launches + 2 syncs per leaf per refresh — measured dominant.
    // out[leaf][b][player][combo].
    void all_leaf_reaches(
        std::vector<std::vector<std::array<std::vector<double>, 2>>>& out,
        int threads = 0);

    // Upload leaf values: CPU tensor [B, 52, 2, n] (chips, absolute — the
    // oracle's ×pot convention), card slot = runout river card; slots for
    // board cards / combos containing the card are ignored by the walk.
    void set_leaf_values(int leaf_idx, const torch::Tensor& vals);

    // ON-DEVICE refresh (f32): reaches, per-runout masked ranges, equity
    // (unit-pot showdown / compat mass via the sorted-prefix structures),
    // percentiles (precomputed per (spec, card)), feature assembly and
    // the net forward all stay on the device — zero PCIe in the refresh
    // loop. Replaces the leaf_reaches → oracle → set_leaf_values
    // round-trip; feature semantics mirror HunlNetOracle's river-query
    // path (raw masked reaches, f32 arithmetic).
    void refresh_leaves_device(HunlValueNet& net, double stack);

    // One CFR iteration (both alternating update passes). Root values of
    // the CURRENT profile accumulate for avg_root_values.
    void iterate(int t);

    // Full solve: refresh(t) fires exactly on the CPU solver's schedule
    // (t == 1 || refresh_every <= 1 || t % refresh_every == 0), then the
    // iteration runs. The callback does the leaf_reaches → oracle →
    // set_leaf_values round-trip.
    void solve(int T, int refresh_every,
               const std::function<void(int)>& refresh);

    // Iterate-averaged root values (ReBeL's training target), same
    // normalization and masking as HunlSolver::avg_root_values.
    bool avg_root_values(
        std::vector<std::array<std::vector<double>, 2>>& v,
        std::vector<std::array<std::vector<double>, 2>>& mask);

    int batch() const { return B_; }
    torch::Dtype dtype() const { return dt_; }
    // final cumulative average-strategy numerators of a decision node,
    // [B, A, n] on device — the state a CPU solver adopts at play time
    const torch::Tensor& cum_state(int node) const {
        return cum_[static_cast<size_t>(node)];
    }

private:
    double weight(int t) const {
        return pcfr_ && pcfr_quad_
            ? static_cast<double>(t) * static_cast<double>(t)
            : static_cast<double>(t);
    }
    void iterate_body();   // both update passes + root-value accumulation
    // persistent fused kernel: device tables + bounds audit (once), then
    // whole refresh windows per launch. -1 = unavailable (fallback).
    bool fused_setup();
    bool fused_window(int t_start, int t_end);
    // equity for every (spec, runout) at once: [B, 52, n] from opp ranges
    // [B, 52, n] (masked, unnormalized — equity is scale-invariant)
    torch::Tensor runout_equity(const torch::Tensor& opp52);
    torch::Tensor policies(int node, bool average);       // [B, A, n]
    torch::Tensor fold_cfv_t(int node, int upd, const torch::Tensor& opp);
    torch::Tensor allin_cfv_t(int node, const torch::Tensor& opp);
    torch::Tensor street_end_cfv_t(int node, int upd,
                                   const torch::Tensor& opp);
    torch::Tensor walk(int node, int upd, bool update, torch::Tensor my_reach,
                       torch::Tensor opp_reach);

    TreeShape shape_;
    torch::Device dev_;
    torch::Dtype dt_;
    int B_ = 0;
    bool pcfr_ = false;
    bool pcfr_quad_ = true;

    // constants
    torch::Tensor inc_;               // [n, 52]
    torch::Tensor card_a_, card_b_;   // [n] long
    torch::Tensor mask_xc_;           // [52, n]: combo x excludes card c
    // per-spec
    torch::Tensor valid_;             // [B, n] (valid on the 4-card board)
    torch::Tensor r0_, r1_;           // [B, n] normalized root ranges
    torch::Tensor r0c_, r1c_;         // CPU float copies (reach walks)
    torch::Tensor runout_ok_;         // [B, 52, 1]: card off the board
    torch::Tensor c0_, c1_;           // [B, nodes] contribs
    // AllinShowdown as a precomputed linear operator: cfv = Op · opp,
    // Op[x][y] = half_pot/44 · Σ_runouts sign(x beats y) with exact card
    // removal — one bmm per pass instead of the 48-runout showdown
    // pipeline (measured dominant per iteration). [B, n, n], ~225MB at
    // B=32 f32. REBEL_NO_ALLIN_OP falls back to the runout kernels.
    bool use_allin_op_ = false;
    torch::Tensor allin_op_;
    // fallback: per-(spec, runout) showdown structure, runouts on the
    // batch dim (built + uploaded only when the operator is off)
    torch::Tensor perm52_, prev52_, seg52_;      // [B*52, n] long
    torch::Tensor poscard52_;                    // [B*52, 52*51] long
    torch::Tensor ilowA52_, ilowB52_, iendA52_, iendB52_, itotA52_,
        itotB52_;                                // [B*52, n] long
    // per-StreetEnd-leaf net values [B, 52, 2, n]
    std::vector<int> leaf_nodes_;
    std::vector<std::vector<int>> leaf_path_;    // env actions root → leaf
    std::vector<torch::Tensor> leaf_v_;
    // per-decision-node learning state [B, A, n]
    std::vector<torch::Tensor> regret_, cum_, pred_;
    torch::Tensor w_dev_;             // averaging weight (t or t²)
    // iterate-averaged root values
    std::array<torch::Tensor, 2> root_acc_{};
    long rv_n_ = 0;
    std::unique_ptr<at::cuda::CUDAGraph> graph_;
    // fused-kernel device tables
    int fused_state_ = 0;   // 0 unknown, 1 ready, -1 unavailable
    torch::Tensor f_kind_, f_actor_, f_arity_, f_cbase_, f_cflat_;
    torch::Tensor f_rptr_, f_cptr_, f_pptr_, f_lptr_, f_idpair_;
    // on-device featurizer constants
    torch::Tensor pct52_;    // [B, 52, n] percentile per (spec, runout)
    torch::Tensor fbase_;    // [B, 52] base-board one-hot
    torch::Tensor eye52_;    // [52, 52]
    torch::Tensor feat_X_;   // [L·B·52, kDim] persistent feature buffer
};

class HunlNetOracle;

// Reusable host buffers for refresh_turn_leaves: the per-refresh feature
// block (~250MB at B=32) and per-leaf value tensors (~200MB) re-alloc +
// first-touch cost more than the math when created fresh every refresh.
struct TurnRefreshWorkspace {
    torch::Tensor X;                   // [R, kDim] float CPU (pinned)
    torch::Tensor Y;                   // [R, 2n] float CPU (forward out)
    std::vector<torch::Tensor> vals;   // per leaf [B, 52, 2, n]
};

// Standard oracle round-trip for one refresh: leaf beliefs via ONE
// policy download + CPU walks, feature rows for every (spec, leaf,
// card) built by `threads` workers into one block, ONE net forward,
// threaded scatter back. One oracle per spec (the per-card strength
// context is keyed by the spec's board; contexts persist across
// refreshes). Values are thread-partition-independent.
void refresh_turn_leaves(
    BatchTurnSolver& s, const std::vector<TurnSpec>& specs,
    const std::vector<HunlNetOracle*>& oracles, int threads = 0,
    TurnRefreshWorkspace* ws = nullptr);

}  // namespace rebel_hunl
