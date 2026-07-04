#!/usr/bin/env bash
#
# bootstrap.sh — provision a fresh GPU VM / container for poker_ppo:
#   1. install the build toolchain (apt, run as root — no sudo needed)
#   2. clone the repo + checkout the training branch
#   3. configure & build WITHOUT tests (training binary + table generator)
#   4. generate HandRanks.dat (the 130 MB 2+2 evaluator table, gitignored)
#
# This is provisioning only — it does NOT start training. Launch with
# start.sh (or the binary directly) afterwards; see the final notes.
#
# Assumes the base image already has CUDA + a Python with the matching
# torch installed (the build links against `python3 -c torch.cmake_prefix_path`).
# A pip torch 2.4.x is fine to BUILD against; for the CUDA-graph speedup at
# runtime see the allocator note printed at the end.
#
# Usage:
#   ./bootstrap.sh                       # defaults below
#   REPO_DIR=/work/poker_ppo BRANCH=main ./bootstrap.sh
#
set -euo pipefail

# ── config (override via env) ───────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/ElijahCra/poker_ppo.git}"
BRANCH="${BRANCH:-training-perf-and-sample-efficiency}"
REPO_DIR="${REPO_DIR:-/poker_ppo}"
BUILD_DIR="${BUILD_DIR:-cmake-build-release}"
JOBS="${JOBS:-$(nproc)}"

echo "==> poker_ppo bootstrap"
echo "    repo   : $REPO_URL @ $BRANCH"
echo "    target : $REPO_DIR   build: $BUILD_DIR   jobs: $JOBS"

# ── 1. toolchain (no sudo: must be root) ────────────────────────────────────
if [ "$(id -u)" -ne 0 ]; then
    echo "error: not root and sudo isn't available — run this as root" >&2
    exit 1
fi
echo "==> installing build toolchain (apt)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates

# cmake must be >= 3.18 (project requirement).
cmake_ver=$(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?')
echo "    cmake $cmake_ver, g++ $(g++ -dumpversion)"

# Soft checks for the build's external deps (not apt-installed here).
command -v nvcc  >/dev/null 2>&1 || echo "    WARN: nvcc not found — CUDA toolkit may be missing"
python3 -c 'import torch; print("    torch", torch.__version__, "cuda", torch.version.cuda)' \
    || echo "    WARN: python3 has no torch — the CMake torch discovery will fail"

# ── 2. clone / update + checkout branch ─────────────────────────────────────
if [ -d "$REPO_DIR/.git" ]; then
    echo "==> repo present, fetching"
    git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO_DIR" checkout -B "$BRANCH" "origin/$BRANCH"
else
    echo "==> cloning"
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
echo "    HEAD: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"

# ── 3. configure + build (no tests) ─────────────────────────────────────────
echo "==> configuring (tests OFF)"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
echo "==> building poker_ppo + generate_table"
cmake --build "$BUILD_DIR" -j "$JOBS" --target poker_ppo generate_table

# ── 4. generate HandRanks.dat (run from repo root → ./HandRanks.dat) ─────────
GEN="$BUILD_DIR/Game/Utility/TwoPlusTwoHandEvaluator/generate_table"
if [ -f HandRanks.dat ]; then
    echo "==> HandRanks.dat already present ($(du -h HandRanks.dat | cut -f1)), skipping"
else
    echo "==> generating HandRanks.dat (~130 MB, ~1 min)"
    "./$GEN"
fi
ls -lh HandRanks.dat

# ── 5. auto-tune for this GPU (TUNE=0 to skip) ──────────────────────────────
# Two dynamics-NEUTRAL knobs, both safe to auto-apply (no BR needed):
#   • rollout shape (num_envs at fixed batch) → baked into config.h + rebuilt
#   • CPU worker count → tune.env (runtime; the all-cores default is past the
#     knee on many-core boxes). Batch scaling / bigger model are NOT here —
#     they change dynamics and must be done deliberately + BR-validated
#     (see tools/scale_config.sh and the notes below).
#if [ "${TUNE:-1}" = "1" ]; then
#    echo "==> auto-tuning rollout shape (bench_throughput.sh apply)"
#    bash tools/bench_throughput.sh apply || \
#        echo "    rollout tuning skipped/failed (non-fatal — default kept)"
#
#    NE=$(sed -n "/static constexpr PPOConfig kPPOConfig/,/^};/{s/.*\.num_envs[[:space:]]*=[[:space:]]*\([0-9][0-9]*\).*/\1/p;}" "$REPO_DIR/include/config.h" | head -1)
#    echo "==> auto-tuning CPU workers at num_envs=$NE"
#    T=$(bash tools/bench_throughput.sh threads "$NE" 20 8 16 32 48 64 96 128 2>/dev/null \
#        | sed -n 's/.*POKER_PPO_STEP_THREADS=\([0-9][0-9]*\).*/\1/p')
#    if [ -n "$T" ]; then
#        echo "export POKER_PPO_STEP_THREADS=$T" > tune.env
#        echo "    best workers=$T → wrote tune.env (source it before training)"
#    else
#        echo "    thread tuning skipped/failed (non-fatal — default all-cores)"
#    fi
#fi

cat <<NOTE

==> bootstrap complete.

Start training (native Linux survives SSH disconnect via tmux):
    cd $REPO_DIR
    tmux new -s train
    [ -f tune.env ] && source tune.env      # POKER_PPO_STEP_THREADS from autotune
    ./$BUILD_DIR/poker_ppo 2>&1 | tee train.log

Bigger GPU (H100/H200): the rollout shape + threads are auto-tuned above,
but those are dynamics-NEUTRAL. To actually spend an H100's headroom, scale
the BATCH (a dynamics change — validate it):
    tools/scale_config.sh 4 --apply         # 4× batch, coherent + rebuild
    POKER_PPO_BR_SEEDS=3 POKER_PPO_MAX_STEPS=37000000 ./$BUILD_DIR/poker_ppo  # BR check
A bigger MODEL (hidden_dim/num_layers/attn_dim) is the other capacity lever
— same: edit config, BR-validate. Neither is auto-applied on purpose.

torch < 2.5 (e.g. 2.4.x) note: the expandable-segments allocator we set
for WSL is NOT CUDA-graph-capture-safe before torch 2.5, and would crash
capture. Disable it (keeps the graph speedup via the native allocator):
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
…or fall back to eager (slower, always safe):
    export POKER_PPO_NO_CUDA_GRAPH=1 POKER_PPO_NO_UPDATE_GRAPH=1

Checkpoints land in runs/<ts>/ckpt; resume with:
    POKER_PPO_RESUME=\$(ls -td runs/*/ckpt | head -1) ./$BUILD_DIR/poker_ppo
NOTE
