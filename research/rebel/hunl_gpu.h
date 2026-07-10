// Batched river subgame solver on torch tensors (ReBeL stage 2, GPU path).
//
// Solves B river subgames in LOCKSTEP: one shared betting-tree topology
// (asserted per batch; sparse-action river trees are near-always identical),
// per-spec boards / pots / ranges. State per decision node is [B, A, 1326];
// CFR+ (RM⁺, alternating updates, linear averaging) — the same math as
// HunlSolver, judged by `gpu_check` equivalence against it.
//
// Kernels:
//   fold      — compatible mass via one matmul with the [1326, 52] card
//               incidence matrix: mass = S − Sc[a] − Sc[b] + w.
//   showdown  — per-row precomputed rank permutation + tie-group segment
//               indices; per call: gather → cumsum (total and per-card) →
//               boundary gathers give strictly-lower / tie / total masses
//               with exact card removal; scatter back.
// Everything is fp32 on `device` (validated ~1e-3 vs the fp64 CPU solver);
// invalid/pad combos carry zero mass throughout and are masked at the end.
//
// This replaces per-target ExactStreetOracle solves in train_river: one
// batched solve produces B training targets.
#pragma once

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "hunl_solver.h"

namespace at::cuda {
struct CUDAGraph;   // avoid pulling CUDA headers into every consumer
}

namespace rebel_hunl {

struct RiverSpec {
    std::array<uint8_t, 5> board{};
    std::vector<double> r0, r1;          // root ranges (unnormalized ok)
    std::vector<int> node_contrib0;      // per tree node, this spec's chips
    std::vector<int> node_contrib1;
};

// Topology extracted from a CPU-built tree (HunlSolver at the river root).
struct TreeShape {
    struct Node {
        int kind;                        // HunlSolver::Node::Kind
        int player;                      // actor (Decision) / folder (Fold)
        std::vector<int> acts;
        std::vector<int> child;
    };
    std::vector<Node> nodes;
    std::string signature;               // grouping key
    static TreeShape from(const HunlSolver& s);
};

class BatchRiverSolver {
public:
    // dtype: kFloat default (GPU-fast; ~0.5%-of-pot accumulation drift over
    // hundreds of iterations — negligible vs net error); kDouble available
    // for exact equivalence checking against the CPU solver.
    BatchRiverSolver(TreeShape shape, std::vector<RiverSpec> specs,
                     torch::Device device,
                     torch::Dtype dtype = torch::kFloat);
    ~BatchRiverSolver();

    void iterate(int t);
    // Full solve with the launch-overhead fix: the two-pass iteration is
    // captured ONCE as a CUDA graph and replayed T times (per iteration:
    // one fill_ of the averaging weight + one graph launch, instead of
    // ~2k kernel launches — the measured reason this solver lost to CPU
    // workers). State updates are in-place on stable storage, so replays
    // accumulate exactly like eager iterates; `gpu_check` verifies the
    // graphed path against both the eager path and the fp64 CPU solver.
    // Falls back to eager iterates off-CUDA, for tiny T, or if capture
    // fails (REBEL_NO_CUDA_GRAPH=1 forces eager).
    void solve(int T);
    // per-spec root values of the average profile, normalized per combo by
    // opponent compatible mass (CPU doubles) + masks — training targets.
    void root_values(
        std::vector<std::array<std::vector<double>, 2>>& v,
        std::vector<std::array<std::vector<double>, 2>>& mask);

    int batch() const { return B_; }

private:
    torch::Tensor walk(int node, int upd, int t, bool update,
                       torch::Tensor my_reach, torch::Tensor opp_reach);
    torch::Tensor policies(int node, bool average);       // [B, A, n]
    torch::Tensor fold_cfv_t(int node, int upd, const torch::Tensor& opp);
    torch::Tensor showdown_cfv_t(int node, const torch::Tensor& opp);

    TreeShape shape_;
    torch::Device dev_;
    torch::Dtype dt_;
    int B_ = 0;

    // constants
    torch::Tensor inc_;        // [n, 52] card incidence (float)
    torch::Tensor card_a_, card_b_;   // [n] long: the two cards per combo
    // per-spec
    torch::Tensor valid_;      // [B, n] float
    torch::Tensor r0_, r1_;    // [B, n] normalized root ranges
    torch::Tensor perm_, prev_end1_, seg_end_;   // [B, n] long (showdown)
    // Sparse per-card showdown structure: each card appears in exactly 51
    // combos, so card-restricted prefix masses live in [B, 52, 51] instead
    // of the old dense [B, n, 52] cumsum (~26× wasted memory traffic —
    // measured kernel-bound at 11 targets/s vs CPU's 81). pos_card_:
    // sorted positions of card c's combos, ascending. idx_*: CONSTANT
    // flattened (card·52 + count-below-boundary) indices into the
    // zero-padded per-card prefix table [B, 52·52] — boundaries are fixed
    // per spec, so the counts precompute on CPU.
    torch::Tensor pos_card_;                     // [B, 52*51] long
    torch::Tensor idx_lowA_, idx_lowB_;          // [B, n] long
    torch::Tensor idx_endA_, idx_endB_;          // [B, n] long
    torch::Tensor idx_totA_, idx_totB_;          // [B, n] long
    torch::Tensor c0_, c1_;    // [B, nodes] float contribs
    // per decision node learning state, [B, A, n]
    std::vector<torch::Tensor> regret_, cum_;
    // linear-averaging weight as a device scalar: filled before each
    // eager iterate / graph replay, read inside the (captured) walk
    torch::Tensor t_dev_;
    std::unique_ptr<at::cuda::CUDAGraph> graph_;
};

}  // namespace rebel_hunl
