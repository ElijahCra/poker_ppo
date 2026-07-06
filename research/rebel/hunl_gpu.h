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
#include <string>
#include <vector>

#include "hunl_solver.h"

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

    void iterate(int t);
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
    torch::Tensor a_sorted_, b_sorted_;          // [B, n] long
    torch::Tensor card_sorted_;                  // [B, n, 52] float (const)
    torch::Tensor c0_, c1_;    // [B, nodes] float contribs
    // per decision node learning state, [B, A, n]
    std::vector<torch::Tensor> regret_, cum_;
};

}  // namespace rebel_hunl
