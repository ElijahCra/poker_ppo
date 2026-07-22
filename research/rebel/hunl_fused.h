// Persistent-kernel river CFR: ONE cuda block per subgame runs the whole
// T-iteration solve in a single kernel launch — no per-op launches, no
// graph replays, all intermediates in registers/shared memory. Compiled
// at runtime via NVRTC (libtorch bundles it; no nvcc / CMake CUDA needed)
// and launched through the driver API. fp32 only; the torch graph path
// remains for f64 checks and as the fallback.
#pragma once

#include <cstdint>

namespace rebel_hunl {

struct FusedRiverArgs {
    int T = 0, M = 0, B = 0;
    // topology, int32 device pointers
    const int32_t* kind = nullptr;        // [M] HunlSolver::Node::Kind
    const int32_t* actor = nullptr;       // [M] actor / folder
    const int32_t* arity = nullptr;       // [M]
    const int32_t* child_base = nullptr;  // [M] offsets into child_flat
    const int32_t* child_flat = nullptr;
    // per-decision-node state buffers, device addresses ([B, A, n] each)
    const int64_t* regret_ptr = nullptr;  // [M]
    const int64_t* cum_ptr = nullptr;     // [M]
    // per-spec data (all device)
    const float* r0 = nullptr;            // [B, n]
    const float* r1 = nullptr;            // [B, n]
    const float* valid = nullptr;         // [B, n]
    const float* c0 = nullptr;            // [B, M]
    const float* c1 = nullptr;            // [B, M]
    const int64_t* perm = nullptr;        // [B, n]
    const int64_t* prev1 = nullptr;       // [B, n]
    const int64_t* segend = nullptr;      // [B, n]
    const int64_t* poscard = nullptr;     // [B, 52*51]
    const int64_t* ilowA = nullptr;       // [B, n] …boundary indices
    const int64_t* ilowB = nullptr;
    const int64_t* iendA = nullptr;
    const int64_t* iendB = nullptr;
    const int64_t* itotA = nullptr;
    const int64_t* itotB = nullptr;
    const int64_t* cardA = nullptr;       // [n]
    const int64_t* cardB = nullptr;       // [n]
};

// false = safely unavailable before launch (no NVRTC/driver, compile
// failure, bounds exceeded, or launch rejected) — caller may fall back to
// the graphed torch path. A post-launch synchronization failure throws,
// because solver state may already be partial. Compile happens once per
// process; failures are logged to stderr once.
bool fused_river_solve(const FusedRiverArgs& a);

// Persistent-kernel TURN CFR: iterations t_start..t_end (one refresh
// window) in a single launch. Leaves: Fold (bins), AllinShowdown via the
// antisymmetric precomputed operator (op[x][y] = −op[y][x], so coalesced
// row reads need no transposed copy), StreetEnd via net-value tensors +
// the closed-form compatible mass (bins + pair-combo lookups). CFR+ or
// PCFR+ (pred buffers, independently selectable linear/quadratic averaging).
// Current-profile root values accumulate with the same averaging weights.
struct FusedTurnArgs {
    int t_start = 1, t_end = 0, M = 0, B = 0, pcfr = 0, quad_avg = 0;
    const int32_t* kind = nullptr;        // [M]
    const int32_t* actor = nullptr;       // [M]
    const int32_t* arity = nullptr;       // [M]
    const int32_t* child_base = nullptr;  // [M]
    const int32_t* child_flat = nullptr;
    const int64_t* regret_ptr = nullptr;  // [M] device addresses
    const int64_t* cum_ptr = nullptr;     // [M]
    const int64_t* pred_ptr = nullptr;    // [M] (used when pcfr)
    const int64_t* leaf_ptr = nullptr;    // [M] ([B,52,2,n] per StreetEnd)
    const float* r0 = nullptr;            // [B, n]
    const float* r1 = nullptr;            // [B, n]
    const float* valid = nullptr;         // [B, n]
    const float* c0 = nullptr;            // [B, M]
    const float* c1 = nullptr;            // [B, M]
    const float* allin_op = nullptr;      // [B, n, n], antisymmetric
    const int32_t* id_pair = nullptr;     // [52*52] combo id of {a,b}
    const int64_t* cardA = nullptr;       // [n]
    const int64_t* cardB = nullptr;       // [n]
    float* root_acc0 = nullptr;           // [B, n]
    float* root_acc1 = nullptr;           // [B, n]
};
bool fused_turn_solve(const FusedTurnArgs& a);

}  // namespace rebel_hunl
