#!/usr/bin/env bash
# Rental-box probe (2026-07-10): one run captures everything needed to
# calibrate the GPU solving/training plan on new hardware — box specs,
# the full validation ladder (Blackwell/new-arch sanity for the NVRTC
# kernel), solver throughput across batch sizes, and end-to-end
# train_river epoch times CPU-only vs GPU-pipelined.
#
# Prerequisites on the rental box:
#   - repo cloned, HandRanks.dat present (repo root or Game/Utility/)
#   - python torch with CUDA (pip install torch) — bundles libnvrtc
#   - built:  cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release \
#               -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
#             cmake --build cmake-build-release --target rebel_hunl -j
#
# Run:  bash tools/rebel_rental_probe.sh
# Send back: rental_probe.log   (~20 min total)
set -u
cd "$(dirname "$0")/.."
BIN=./cmake-build-release/rebel_hunl
OUT=rental_probe.log
NP=$(nproc)

{
echo "==================== box ===================="
date
echo "nproc: $NP"
free -g | head -2
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm \
    --format=csv 2>/dev/null
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda)' 2>/dev/null

echo "==================== validation ===================="
# kernels: CPU fast-vs-brute; gpu_check: eager + fused NVRTC kernel vs
# fp64 CPU solver (first Blackwell run of the runtime-compiled kernel);
# gadget_check: safe-resolving guarantee
$BIN kernels
$BIN gpu_check 16 200
REBEL_GPU_F64=1 $BIN gpu_check 16 200
$BIN gadget_check 400

echo "==================== solver bench ===================="
# per-B: CPU worker pool vs fused-kernel GPU, targets/s
for B in 512 2048 8192 16384; do
    $BIN gpu_bench "$B" 200 "$NP" || echo "gpu_bench B=$B failed (VRAM?)"
done

echo "==================== end-to-end train_river ===================="
echo "--- CPU-only (2 epochs x 4000):"
REBEL_SEED=3 REBEL_CKPT= REBEL_PROBE_K=0 REBEL_HIDDEN=64 \
    REBEL_SGD_STEPS=5 $BIN train_river 2 4000 2>&1 | grep -E 'HUNL|epoch'
echo "--- pipelined, GPU-heavy split (2 epochs x 4000):"
REBEL_SEED=3 REBEL_CKPT= REBEL_PROBE_K=0 REBEL_HIDDEN=64 \
    REBEL_SGD_STEPS=5 REBEL_GPU_BATCH=4096 REBEL_GPU_FRAC=0.8 \
    $BIN train_river 2 4000 2>&1 | grep -E 'HUNL|epoch'
echo "--- pipelined, balanced split:"
REBEL_SEED=3 REBEL_CKPT= REBEL_PROBE_K=0 REBEL_HIDDEN=64 \
    REBEL_SGD_STEPS=5 REBEL_GPU_BATCH=4096 REBEL_GPU_FRAC=0.5 \
    $BIN train_river 2 4000 2>&1 | grep -E 'HUNL|epoch'

echo "==================== done ===================="
date
} 2>&1 | tee "$OUT"

echo
echo ">>> send back: $OUT"
