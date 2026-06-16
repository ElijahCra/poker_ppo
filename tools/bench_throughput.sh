#!/usr/bin/env bash
# bench_throughput.sh — find the rollout-throughput knee (best num_envs) on
# this machine's GPU. No rebuild needed.
#
# POKER_PPO_NUM_ENVS overrides num_envs for --benchmark ONLY. The benchmark
# times the rollout (collect()) — batched inference + env stepping — at each
# num_envs and reports microseconds per sample. As num_envs grows, per-sample
# cost falls (launch latency amortises over a bigger inference batch) until it
# flattens: that flat point is the "knee". A faster GPU hides more launches,
# so its knee sits at a HIGHER num_envs than the 3060 Ti's ~384.
#
# This measures the ROLLOUT half only — the update() path uses the compiled
# cfg_.num_envs and is unaffected by the override (that's why it's safe here
# but must not be used to train). For the update half, see the block at the
# bottom of this file.
#
# ── CPU thread sweep (NEW) ──────────────────────────────────────────────────
# env_step (game engine + obs build) is the CPU-bound rollout phase; the step
# thread pool parallelises it across envs. POKER_PPO_STEP_THREADS overrides
# the worker count (default = all cores, capped at num_envs). Also runtime,
# so this sweep needs no rebuild. More threads help only up to a knee — each
# env step is tiny, so past it the per-step barrier sync dominates. Run this
# to size the pool on a many-core box (the default of "all cores" can be past
# the knee → oversubscription).
#
# Usage:
#   tools/bench_throughput.sh [binary] [iters] [N1 N2 ...]          # num_envs sweep (GPU)
#   tools/bench_throughput.sh threads [num_envs] [iters] [T1 T2 ...] # thread sweep (CPU)
#   tools/bench_throughput.sh                       # defaults
#   tools/bench_throughput.sh cmake-build-release/poker_ppo 40 512 1024 2048 4096
#   tools/bench_throughput.sh threads 1024 30 4 8 16 24 32 48 64
set -uo pipefail

# Thread-sweep mode dispatches early; everything below is the num_envs sweep.
if [ "${1:-}" = "threads" ]; then
  BIN="cmake-build-release/poker_ppo"
  NENVS="${2:-1024}"
  ITERS="${3:-30}"
  THREADS=("${@:4}")
  if [ ${#THREADS[@]} -eq 0 ]; then THREADS=(2 4 8 16 24 32 48 64); fi
  if [ ! -x "$BIN" ]; then echo "binary not found: $BIN" >&2; exit 1; fi
  echo "CPU thread sweep at num_envs=$NENVS ($(nproc) logical cores available)"
  printf '%9s  %14s  %12s\n' threads "us/sample" "rollout_ms"
  printf '%9s  %14s  %12s\n' "-------" "---------" "----------"
  bu=""; bt=""
  for T in "${THREADS[@]}"; do
    [ "$T" -gt "$NENVS" ] 2>/dev/null && continue   # capped at num_envs anyway
    line=$(POKER_PPO_STEP_THREADS="$T" POKER_PPO_NUM_ENVS="$NENVS" \
           "$BIN" --benchmark "$ITERS" 2>/dev/null \
           | grep -E '^[[:space:]]+threadpool' | head -1 || true)
    us=$(sed -n 's/.*us\/samp=\([0-9.][0-9.]*\).*/\1/p' <<<"$line")
    ms=$(sed -n 's/.*med=\([0-9.][0-9.]*\)ms.*/\1/p' <<<"$line")
    if [ -z "$us" ]; then printf '%9s  %14s\n' "$T" "ERR"; continue; fi
    printf '%9s  %14s  %12s\n' "$T" "$us" "$ms"
    if [ -z "$bu" ] || awk "BEGIN{exit !($us < $bu)}"; then bu="$us"; bt="$T"; fi
  done
  echo
  echo "Best: POKER_PPO_STEP_THREADS=$bt at ${bu} us/sample."
  echo "Adopt the smallest thread count within ~5% of best (lower = less"
  echo "contention with the GPU/driver threads). Default (all cores) is good"
  echo "if it's at/below the knee; if a smaller count is faster, the pool is"
  echo "oversubscribed — set POKER_PPO_STEP_THREADS at runtime, or change the"
  echo "default in RolloutCollector::ensure_step_pool (src/rollout.cpp)."
  exit 0
fi

BIN="${1:-cmake-build-release/poker_ppo}"
ITERS="${2:-30}"
ENVS=("${@:3}")
if [ ${#ENVS[@]} -eq 0 ]; then
  ENVS=(192 384 512 768 1024 1536 2048 3072 4096)
fi

if [ ! -x "$BIN" ]; then echo "binary not found/executable: $BIN" >&2; exit 1; fi

printf '%9s  %14s  %12s  %12s\n' num_envs "us/sample" "rollout_ms" "samples/s"
printf '%9s  %14s  %12s  %12s\n' "--------" "---------" "----------" "---------"
best_us=""; best_n=""
for N in "${ENVS[@]}"; do
  # 3 warmup iters are baked into --benchmark; threadpool row is the one we use.
  # Match the data row (leading whitespace + "threadpool"), NOT the
  # "comparing serial / threadpool" header line.
  line=$(POKER_PPO_NUM_ENVS="$N" "$BIN" --benchmark "$ITERS" 2>/dev/null \
         | grep -E '^[[:space:]]+threadpool' | head -1 || true)
  us=$(sed -n 's/.*us\/samp=\([0-9.][0-9.]*\).*/\1/p' <<<"$line")
  ms=$(sed -n 's/.*med=\([0-9.][0-9.]*\)ms.*/\1/p' <<<"$line")
  if [ -z "$us" ]; then
    printf '%9s  %14s  %12s  %12s\n' "$N" "ERR/OOM" "-" "-"
    continue
  fi
  sps=$(awk "BEGIN{printf \"%.0f\", 1e6/$us}")
  printf '%9s  %14s  %12s  %12s\n' "$N" "$us" "$ms" "$sps"
  if [ -z "$best_us" ] || awk "BEGIN{exit !($us < $best_us)}"; then
    best_us="$us"; best_n="$N"
  fi
done

echo
echo "Lowest us/sample: num_envs=$best_n at ${best_us} us/sample."
echo "Pick the knee: the smallest num_envs whose us/sample is within ~5% of"
echo "this best — past it you pay VRAM/latency for negligible throughput."
echo
cat <<'NOTE'
─── after choosing num_envs ────────────────────────────────────────────────
This is a COMPILE-TIME change (config.h), then rebuild:

  config::kPPOConfig.num_envs        = <chosen N>
  config::kPPOConfig.num_minibatches = <N * 128 / 3072>   # hold minibatch=3072

Holding minibatch_size at 3072 keeps training dynamics identical (gradient
noise scale, optimizer steps/sample, the captured update-graph shape) — only
the speed changes. Keep all cadences as-is; they're in env steps already.

─── update-side check (the half --benchmark can't see) ─────────────────────
The update is GPU-compute-bound, so a faster GPU helps it directly, and a
bigger GPU may prefer a LARGER minibatch (better GEMM utilisation) — but that
changes learning, not just speed, so validate it with a BR A/B, don't just
chase ms. To time end-to-end at a chosen config, rebuild then:

  POKER_PPO_PROFILE=1 POKER_PPO_MAX_STEPS=3000000 ./cmake-build-release/poker_ppo \
    2>&1 | grep -E 'update phase profile' -A 9

Read the steady-state "[update K] rollout=.. update=.." lines (ignore update 0,
which includes CUDA-graph capture) and the printed update-phase breakdown.
samples/sec end-to-end = num_envs*128 / (rollout_ms + update_ms) * 1000.
NOTE
