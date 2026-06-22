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
#   tools/bench_throughput.sh [binary] [iters] [N1 N2 ...]            # num_envs sweep (GPU)
#   tools/bench_throughput.sh threads [num_envs] [iters] [T1 T2 ...]  # thread sweep (CPU)
#   tools/bench_throughput.sh grid [iters] "<envs...>" "<threads...>" # joint num_envs×threads
#   tools/bench_throughput.sh apply [iters]                          # pick knee → edit config → rebuild
#   tools/bench_throughput.sh                       # defaults
#   tools/bench_throughput.sh cmake-build-release/poker_ppo 40 512 1024 2048 4096
#   tools/bench_throughput.sh threads 1024 30 4 8 16 24 32 48 64
#   tools/bench_throughput.sh grid 20 "768 1024 1536" "16 32 48 64"
set -uo pipefail

# ── apply mode: sweep the rollout knee and rewrite config.h, then rebuild ───
# Dynamics-neutral: holds batch_size fixed (minibatch, gradient steps, and
# every validated result unchanged) by trading num_steps for num_envs.
# Floors num_steps (NSTEPS_FLOOR, default 32) so the GAE horizon stays sane —
# going below that, or changing batch_size/minibatch, needs a BR A/B and is
# left to tune_speed.sh + manual review. Usage: bench_throughput.sh apply [iters]
if [ "${1:-}" = "apply" ]; then
  BUILD_DIR="${BUILD_DIR:-cmake-build-release}"
  BIN="$BUILD_DIR/poker_ppo"
  CFG="include/config.h"
  ITERS="${2:-30}"
  TOL="${KNEE_TOL:-0.05}"
  NSTEPS_FLOOR="${NSTEPS_FLOOR:-32}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
  [ -x "$BIN" ] || { echo "binary not found: $BIN" >&2; exit 1; }
  [ -f "$CFG" ] || { echo "config not found: $CFG (run from repo root)" >&2; exit 1; }

  read_cfg() {  # value of a kPPOConfig field (kBRConfig protected by the range)
    sed -n "/static constexpr PPOConfig kPPOConfig/,/^};/{s/.*\.$1[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p;}" "$CFG" | head -1
  }
  cur_envs=$(read_cfg num_envs); cur_steps=$(read_cfg num_steps)
  batch=$(( cur_envs * cur_steps ))
  echo "current: num_envs=$cur_envs num_steps=$cur_steps batch=$batch (held fixed)"
  echo "sweeping batch-invariant num_envs (num_steps=batch/num_envs >= $NSTEPS_FLOOR):"

  best_us=""; declare -a CN CU
  for E in 256 512 1024 2048 4096 8192 16384; do
    (( batch % E == 0 )) || continue
    S=$(( batch / E )); (( S >= NSTEPS_FLOOR )) || continue
    line=$(POKER_PPO_NUM_ENVS="$E" "$BIN" --benchmark "$ITERS" 2>/dev/null \
           | grep -E '^[[:space:]]+threadpool' | head -1 || true)
    us=$(sed -n 's/.*us\/samp=\([0-9.][0-9.]*\).*/\1/p' <<<"$line")
    if [ -z "$us" ]; then echo "  envs=$E steps=$S  ERR/OOM"; continue; fi
    printf '  envs=%-5s steps=%-4s us/sample=%s\n' "$E" "$S" "$us"
    CN+=("$E"); CU+=("$us")
    if [ -z "$best_us" ] || awk "BEGIN{exit !($us<$best_us)}"; then best_us="$us"; fi
  done
  [ -n "$best_us" ] || { echo "no candidates measured (OOM?)" >&2; exit 1; }

  # Knee = smallest num_envs within TOL of the best us/sample.
  thresh=$(awk "BEGIN{print $best_us*(1+$TOL)}")
  knee=""
  for i in "${!CN[@]}"; do
    if awk "BEGIN{exit !(${CU[$i]}<=$thresh)}"; then knee=${CN[$i]}; break; fi
  done
  S=$(( batch / knee ))
  echo "knee (within $(awk "BEGIN{print $TOL*100}")% of best): num_envs=$knee num_steps=$S"
  if [ "$knee" = "$cur_envs" ]; then echo "already at the knee — no change."; exit 0; fi

  # Edit only num_envs/num_steps in kPPOConfig (batch & minibatch unchanged).
  sed -i "/static constexpr PPOConfig kPPOConfig/,/^};/{
    s/\(\.num_envs[[:space:]]*=[[:space:]]*\)[0-9][0-9]*/\1$knee/
    s/\(\.num_steps[[:space:]]*=[[:space:]]*\)[0-9][0-9]*/\1$S/
  }" "$CFG"
  echo "applied → $CFG; rebuilding poker_ppo..."
  cmake --build "$BUILD_DIR" -j --target poker_ppo
  echo "done: num_envs=$knee num_steps=$S (batch $batch, minibatch unchanged)"
  echo "tip: also sweep CPU workers — bench_throughput.sh threads $knee"
  exit 0
fi

# Thread-sweep mode dispatches early; everything below is the num_envs sweep.
if [ "${1:-}" = "threads" ]; then
  BIN="cmake-build-release/poker_ppo"
  NENVS="${2:-1024}"
  ITERS="${3:-30}"
  THREADS=("${@:4}")
  if [ ${#THREADS[@]} -eq 0 ]; then THREADS=(8 16 32 64 128); fi
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

# ── grid mode: joint num_envs × threads sweep (no rebuild) ──────────────────
# The CPU-worker knee shifts with num_envs (more envs = more parallel env
# steps to fill cores), so the two interact — this sweeps both at once and
# prints a us/sample matrix (rows = num_envs, cols = threads; lower = better).
# Usage: bench_throughput.sh grid [iters] "<envs...>" "<threads...>"
if [ "${1:-}" = "grid" ]; then
  BUILD_DIR="${BUILD_DIR:-cmake-build-release}"
  BIN="$BUILD_DIR/poker_ppo"
  ITERS="${2:-20}"
  read -r -a G_ENVS    <<<"${3:-384 768 1024 1536}"
  read -r -a G_THREADS <<<"${4:-8 16 32 48}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:256}"
  [ -x "$BIN" ] || { echo "binary not found: $BIN" >&2; exit 1; }
  echo "joint sweep: us/sample (lower=better), $(nproc) cores, iters=$ITERS"

  printf '%9s' 'envs\thr'
  for T in "${G_THREADS[@]}"; do printf '%9s' "$T"; done; printf '\n'
  best_us=""; best_cell=""
  for E in "${G_ENVS[@]}"; do
    printf '%9s' "$E"
    for T in "${G_THREADS[@]}"; do
      if [ "$T" -gt "$E" ] 2>/dev/null; then printf '%9s' '·'; continue; fi
      line=$(POKER_PPO_NUM_ENVS="$E" POKER_PPO_STEP_THREADS="$T" \
             "$BIN" --benchmark "$ITERS" 2>/dev/null \
             | grep -E '^[[:space:]]+threadpool' | head -1 || true)
      us=$(sed -n 's/.*us\/samp=\([0-9.][0-9.]*\).*/\1/p' <<<"$line")
      if [ -z "$us" ]; then printf '%9s' 'ERR'; continue; fi
      printf '%9s' "$us"
      if [ -z "$best_us" ] || awk "BEGIN{exit !($us<$best_us)}"; then
        best_us="$us"; best_cell="num_envs=$E threads=$T"
      fi
    done
    printf '\n'
  done
  echo
  echo "Best: $best_cell at ${best_us} us/sample"
  echo "Rollout-only (the update is unaffected by these). Apply num_envs at the"
  echo "dynamics-neutral knee with 'bench_throughput.sh apply'; set the threads"
  echo "via POKER_PPO_STEP_THREADS=<best> at runtime."
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
