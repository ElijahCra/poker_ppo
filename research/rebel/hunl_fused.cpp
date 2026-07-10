// NVRTC-compiled persistent river CFR kernel (see hunl_fused.h).
//
// The kernel mirrors HunlSolver's iteration exactly: alternating updates,
// RM⁺ with the ((r + v_child) − v_parent) association, linear averaging,
// σ stashed per node at first visit so provisional regret writes never
// pollute descents. Leaves reuse the SAME precomputed sparse showdown
// structure as the torch path (perm / boundary-count indices), evaluated
// with block-wide shared-memory scans. Judged by gpu_check against the
// fp64 CPU solver like every other solver in this repo.
#include "hunl_fused.h"

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace rebel_hunl {

namespace {

// ── minimal NVRTC / CUDA-driver surfaces via dlopen (no toolkit) ────────
using nvrtcProgram = void*;
using CUmodule = void*;
using CUfunction = void*;

struct Api {
    // nvrtc
    int (*create)(nvrtcProgram*, const char*, const char*, int,
                  const char* const*, const char* const*) = nullptr;
    int (*compile)(nvrtcProgram, int, const char* const*) = nullptr;
    int (*ptx_size)(nvrtcProgram, size_t*) = nullptr;
    int (*ptx)(nvrtcProgram, char*) = nullptr;
    int (*log_size)(nvrtcProgram, size_t*) = nullptr;
    int (*log)(nvrtcProgram, char*) = nullptr;
    // driver
    int (*mod_load)(CUmodule*, const void*) = nullptr;
    int (*mod_func)(CUfunction*, CUmodule, const char*) = nullptr;
    int (*launch)(CUfunction, unsigned, unsigned, unsigned, unsigned,
                  unsigned, unsigned, unsigned, void*, void**, void**) =
        nullptr;
    int (*ctx_sync)() = nullptr;
    bool ok = false;
};

Api load_api() {
    Api a;
    void* nv = dlopen("libnvrtc.so.12", RTLD_NOW | RTLD_GLOBAL);
    if (!nv && std::getenv("REBEL_NVRTC"))
        nv = dlopen(std::getenv("REBEL_NVRTC"), RTLD_NOW | RTLD_GLOBAL);
    if (!nv) {
        // pip-installed torch keeps nvrtc in the nvidia/ sibling package
        void* h = dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_NOLOAD);
        (void)h;
        const char* home = std::getenv("HOME");
        if (home) {
            const std::string p =
                std::string(home) +
                "/.local/lib/python3.10/site-packages/nvidia/cuda_nvrtc/"
                "lib/libnvrtc.so.12";
            nv = dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL);
        }
    }
    void* cu = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!nv || !cu) return a;
    auto g = [](void* h, const char* s) { return dlsym(h, s); };
    a.create = reinterpret_cast<decltype(a.create)>(
        g(nv, "nvrtcCreateProgram"));
    a.compile = reinterpret_cast<decltype(a.compile)>(
        g(nv, "nvrtcCompileProgram"));
    a.ptx_size = reinterpret_cast<decltype(a.ptx_size)>(
        g(nv, "nvrtcGetPTXSize"));
    a.ptx = reinterpret_cast<decltype(a.ptx)>(g(nv, "nvrtcGetPTX"));
    a.log_size = reinterpret_cast<decltype(a.log_size)>(
        g(nv, "nvrtcGetProgramLogSize"));
    a.log = reinterpret_cast<decltype(a.log)>(g(nv, "nvrtcGetProgramLog"));
    a.mod_load = reinterpret_cast<decltype(a.mod_load)>(
        g(cu, "cuModuleLoadData"));
    a.mod_func = reinterpret_cast<decltype(a.mod_func)>(
        g(cu, "cuModuleGetFunction"));
    a.launch = reinterpret_cast<decltype(a.launch)>(
        g(cu, "cuLaunchKernel"));
    a.ctx_sync = reinterpret_cast<decltype(a.ctx_sync)>(
        g(cu, "cuCtxSynchronize"));
    a.ok = a.create && a.compile && a.ptx_size && a.ptx && a.log_size &&
           a.log && a.mod_load && a.mod_func && a.launch && a.ctx_sync;
    return a;
}

// ── the kernel ──────────────────────────────────────────────────────────
const char* kSrc = R"CU(
#define N 1326
#define NT 256
#define PC 6
#define MAXD 12
#define MAXA 8
#define CHUNK 6   // ceil(N/NT) contiguous scan chunk

extern "C" __global__ void cfr_river(
    int T, int M,
    const int* kind, const int* actor, const int* arity,
    const int* childBase, const int* childFlat,
    const long long* regretPtr, const long long* cumPtr,
    const float* r0g, const float* r1g, const float* validg,
    const float* c0g, const float* c1g,
    const long long* permg, const long long* prev1g,
    const long long* segendg, const long long* poscardg,
    const long long* ilowAg, const long long* ilowBg,
    const long long* iendAg, const long long* iendBg,
    const long long* itotAg, const long long* itotBg,
    const long long* cardAg, const long long* cardBg) {
  const int b = blockIdx.x, tid = threadIdx.x;
  __shared__ float shA[N + 1];        // staging / exclusive scan cs0
  __shared__ float shB[N];            // ws by sorted position
  __shared__ float shCard[52 * 52];   // per-card zero-padded prefixes
  __shared__ float shBins[56];        // S + per-card fold sums
  __shared__ float shPart[NT];        // chunk totals for the block scan

  const float* R0 = r0g + (size_t)b * N;
  const float* R1 = r1g + (size_t)b * N;
  const float* VAL = validg + (size_t)b * N;
  const float* C0 = c0g + (size_t)b * M;
  const float* C1 = c1g + (size_t)b * M;
  const long long* PERM = permg + (size_t)b * N;
  const long long* PREV1 = prev1g + (size_t)b * N;
  const long long* SEGEND = segendg + (size_t)b * N;
  const long long* POSC = poscardg + (size_t)b * 52 * 51;
  const long long* ILA = ilowAg + (size_t)b * N;
  const long long* ILB = ilowBg + (size_t)b * N;
  const long long* IEA = iendAg + (size_t)b * N;
  const long long* IEB = iendBg + (size_t)b * N;
  const long long* ITA = itotAg + (size_t)b * N;
  const long long* ITB = itotBg + (size_t)b * N;

  float reach[2][MAXD][PC];
  float sig[MAXD][MAXA][PC];
  float vacc[MAXD][PC];
  int ndstk[MAXD];
  int cur[MAXD];

  for (int t = 1; t <= T; ++t)
  for (int upd = 0; upd < 2; ++upd) {
    for (int j = 0; j < PC; ++j) {
      const int i = tid + j * NT;
      if (i < N) { reach[0][0][j] = R0[i]; reach[1][0][j] = R1[i]; }
      vacc[0][j] = 0.f;
    }
    int d = 0; ndstk[0] = 0; cur[0] = -1;
    while (d >= 0) {
      const int m = ndstk[d];
      const int K = kind[m];
      bool leaf_done = false;
      if (K == 1) {                       // Fold
        for (int q = tid; q < 56; q += NT) shBins[q] = 0.f;
        __syncthreads();
        for (int j = 0; j < PC; ++j) {
          const int i = tid + j * NT; if (i >= N) continue;
          const float o = reach[1 - upd][d][j];
          if (o != 0.f) {
            atomicAdd(&shBins[52], o);
            atomicAdd(&shBins[(int)cardAg[i]], o);
            atomicAdd(&shBins[(int)cardBg[i]], o);
          }
        }
        __syncthreads();
        const float u = (upd == actor[m])
            ? -(upd == 0 ? C0[m] : C1[m])
            : (upd == 0 ? C1[m] : C0[m]);
        for (int j = 0; j < PC; ++j) {
          const int i = tid + j * NT; if (i >= N) { vacc[d][j] = 0.f; continue; }
          const float o = reach[1 - upd][d][j];
          vacc[d][j] = u *
              (shBins[52] - shBins[(int)cardAg[i]] -
               shBins[(int)cardBg[i]] + o) * VAL[i];
        }
        __syncthreads();
        leaf_done = true;
      } else if (K == 2 || K == 3) {      // Showdown / AllinShowdown@river
        for (int j = 0; j < PC; ++j) {
          const int i = tid + j * NT;
          if (i < N) shA[i] = reach[1 - upd][d][j];
        }
        __syncthreads();
        for (int j = 0; j < PC; ++j) {
          const int p = tid + j * NT;
          if (p < N) shB[p] = shA[(int)PERM[p]];
        }
        __syncthreads();
        // exclusive block scan of shB into shA (cs0), shA[N] = total.
        {
          const int base = tid * CHUNK;
          float run = 0.f;
          float loc[CHUNK];
          for (int k = 0; k < CHUNK; ++k) {
            const int p = base + k;
            loc[k] = run;
            if (p < N) run += shB[p];
          }
          shPart[tid] = run;
          __syncthreads();
          // Hillis–Steele exclusive scan of the 256 chunk totals
          float x = shPart[tid];
          for (int off = 1; off < NT; off <<= 1) {
            const float y = (tid >= off) ? shPart[tid - off] : 0.f;
            __syncthreads();
            shPart[tid] = x = x + y;
            __syncthreads();
          }
          const float pre = (tid > 0) ? shPart[tid - 1] : 0.f;
          for (int k = 0; k < CHUNK; ++k) {
            const int p = base + k;
            if (p <= N) shA[p] = pre + loc[k];
          }
          if (tid == NT - 1) shA[N] = pre + run;
          __syncthreads();
        }
        // per-card zero-padded prefixes: one thread per card
        if (tid < 52) {
          float run = 0.f;
          const long long* pl = POSC + tid * 51;
          for (int k = 0; k < 51; ++k) {
            shCard[tid * 52 + k] = run;
            run += shB[(int)pl[k]];
          }
          shCard[tid * 52 + 51] = run;
        }
        __syncthreads();
        const float hp = C0[m];
        float out[PC];
        for (int j = 0; j < PC; ++j) {
          const int p = tid + j * NT;
          if (p >= N) { out[j] = 0.f; continue; }
          const float ws = shB[p];
          const float lowT = shA[(int)PREV1[p]];
          const float lowA = shCard[(int)ILA[p]];
          const float lowB = shCard[(int)ILB[p]];
          const float win = lowT - lowA - lowB;
          const float endT = shA[(int)SEGEND[p]];
          const float endA = shCard[(int)IEA[p]];
          const float endB = shCard[(int)IEB[p]];
          const float tie =
              (endT - lowT) - (endA - lowA) - (endB - lowB) + ws;
          const float tot = shA[N] - shCard[(int)ITA[p]] -
                            shCard[(int)ITB[p]] + ws;
          out[j] = hp * (2.f * win + tie - tot);
        }
        __syncthreads();                   // everyone done reading cs0
        for (int j = 0; j < PC; ++j) {
          const int p = tid + j * NT;
          if (p < N) shA[(int)PERM[p]] = out[j];   // scatter by combo
        }
        __syncthreads();
        for (int j = 0; j < PC; ++j) {
          const int i = tid + j * NT;
          vacc[d][j] = (i < N) ? shA[i] * VAL[i] : 0.f;
        }
        __syncthreads();
        leaf_done = true;
      } else if (K == 0) {                // Decision
        const int A = arity[m];
        if (cur[d] < 0) {
          // first visit: σ for all actions from CLEAN regrets + linear
          // averaging into cum (my_reach fixed for this node this pass)
          const float* RG =
              (const float*)regretPtr[m] + (size_t)b * A * N;
          for (int j = 0; j < PC; ++j) {
            const int i = tid + j * NT; if (i >= N) continue;
            float s = 0.f, pk[MAXA];
            for (int k = 0; k < A; ++k) {
              float x = RG[(size_t)k * N + i];
              x = x > 0.f ? x : 0.f;
              pk[k] = x; s += x;
            }
            if (s > 1e-12f)
              for (int k = 0; k < A; ++k) sig[d][k][j] = pk[k] / s;
            else {
              const float u = 1.f / A;
              for (int k = 0; k < A; ++k) sig[d][k][j] = u;
            }
            vacc[d][j] = 0.f;
          }
          if (actor[m] == upd) {
            float* CM = (float*)cumPtr[m] + (size_t)b * A * N;
            for (int j = 0; j < PC; ++j) {
              const int i = tid + j * NT; if (i >= N) continue;
              const float w = (float)t * reach[upd][d][j];
              for (int k = 0; k < A; ++k)
                CM[(size_t)k * N + i] += w * sig[d][k][j];
            }
          }
          cur[d] = 0;
        }
        if (cur[d] < A) {
          const int k = cur[d]++;
          const int a = actor[m];
          for (int j = 0; j < PC; ++j) {
            reach[a][d + 1][j] = reach[a][d][j] * sig[d][k][j];
            reach[1 - a][d + 1][j] = reach[1 - a][d][j];
            vacc[d + 1][j] = 0.f;
          }
          ndstk[d + 1] = childFlat[childBase[m] + k];
          cur[d + 1] = -1;
          ++d;
          continue;
        }
        // children done: regret finalization ((r + v_k) − v_p, clamped)
        if (actor[m] == upd) {
          float* RG = (float*)regretPtr[m] + (size_t)b * A * N;
          for (int j = 0; j < PC; ++j) {
            const int i = tid + j * NT; if (i >= N) continue;
            const float vp = vacc[d][j];
            for (int k = 0; k < A; ++k) {
              const float r = RG[(size_t)k * N + i] - vp;
              RG[(size_t)k * N + i] = r > 0.f ? r : 0.f;
            }
          }
        }
        leaf_done = true;                 // treat as "value ready": ascend
      } else {
        leaf_done = true;                 // StreetEnd never occurs (river)
      }
      if (leaf_done) {
        if (d == 0) { d = -1; break; }
        const int mp = ndstk[d - 1];
        const int ke = cur[d - 1] - 1;
        if (actor[mp] == upd) {
          float* RG =
              (float*)regretPtr[mp] + (size_t)b * arity[mp] * N;
          for (int j = 0; j < PC; ++j) {
            const int i = tid + j * NT; if (i >= N) continue;
            const float vc = vacc[d][j];
            vacc[d - 1][j] += sig[d - 1][ke][j] * vc;
            RG[(size_t)ke * N + i] += vc;    // provisional (σ pre-stashed)
          }
        } else {
          for (int j = 0; j < PC; ++j) vacc[d - 1][j] += vacc[d][j];
        }
        --d;
      }
    }
  }
}
)CU";

struct Kernel {
    Api api;
    CUfunction fn = nullptr;
    bool tried = false;
};
Kernel k_;

bool build_kernel() {
    if (k_.tried) return k_.fn != nullptr;
    k_.tried = true;
    k_.api = load_api();
    if (!k_.api.ok) {
        std::fprintf(stderr, "[fused] NVRTC/driver unavailable — torch "
                     "graph path stays\n");
        return false;
    }
    nvrtcProgram prog = nullptr;
    if (k_.api.create(&prog, kSrc, "cfr_river.cu", 0, nullptr, nullptr))
        return false;
    const char* opts[] = {"--gpu-architecture=compute_86",
                          "--use_fast_math"};
    const int rc = k_.api.compile(prog, 2, opts);
    if (rc != 0) {
        size_t ls = 0;
        k_.api.log_size(prog, &ls);
        std::string log(ls, '\0');
        k_.api.log(prog, log.data());
        std::fprintf(stderr, "[fused] NVRTC compile failed:\n%s\n",
                     log.c_str());
        return false;
    }
    size_t ps = 0;
    k_.api.ptx_size(prog, &ps);
    std::string ptx(ps, '\0');
    k_.api.ptx(prog, ptx.data());
    CUmodule mod = nullptr;
    if (k_.api.mod_load(&mod, ptx.c_str())) {
        std::fprintf(stderr, "[fused] cuModuleLoadData failed\n");
        return false;
    }
    if (k_.api.mod_func(&k_.fn, mod, "cfr_river")) {
        std::fprintf(stderr, "[fused] kernel symbol missing\n");
        k_.fn = nullptr;
        return false;
    }
    return true;
}

}  // namespace

bool fused_river_solve(const FusedRiverArgs& a) {
    if (!build_kernel()) return false;
    void* args[] = {
        const_cast<int*>(&a.T), const_cast<int*>(&a.M),
        const_cast<int32_t**>(&a.kind), const_cast<int32_t**>(&a.actor),
        const_cast<int32_t**>(&a.arity),
        const_cast<int32_t**>(&a.child_base),
        const_cast<int32_t**>(&a.child_flat),
        const_cast<int64_t**>(&a.regret_ptr),
        const_cast<int64_t**>(&a.cum_ptr),
        const_cast<float**>(&a.r0), const_cast<float**>(&a.r1),
        const_cast<float**>(&a.valid), const_cast<float**>(&a.c0),
        const_cast<float**>(&a.c1), const_cast<int64_t**>(&a.perm),
        const_cast<int64_t**>(&a.prev1), const_cast<int64_t**>(&a.segend),
        const_cast<int64_t**>(&a.poscard), const_cast<int64_t**>(&a.ilowA),
        const_cast<int64_t**>(&a.ilowB), const_cast<int64_t**>(&a.iendA),
        const_cast<int64_t**>(&a.iendB), const_cast<int64_t**>(&a.itotA),
        const_cast<int64_t**>(&a.itotB), const_cast<int64_t**>(&a.cardA),
        const_cast<int64_t**>(&a.cardB)};
    const int rc = k_.api.launch(
        k_.fn, static_cast<unsigned>(a.B), 1, 1, 256, 1, 1, 0, nullptr,
        args, nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "[fused] cuLaunchKernel rc=%d\n", rc);
        return false;
    }
    return k_.api.ctx_sync() == 0;
}

}  // namespace rebel_hunl
