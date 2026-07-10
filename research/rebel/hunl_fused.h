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

// false = unavailable (no NVRTC/driver, compile failure, bounds exceeded)
// — caller falls back to the graphed torch path. Compile happens once per
// process; failures are logged to stderr once.
bool fused_river_solve(const FusedRiverArgs& a);

}  // namespace rebel_hunl
