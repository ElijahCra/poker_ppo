#!/usr/bin/env bash
# Rental-box probe (2026-07-10): one run captures everything needed to
# calibrate the GPU solving/training plan on new hardware — box specs,
# the full validation ladder (Blackwell/new-arch sanity for the NVRTC
# kernel), solver throughput across batch sizes, and end-to-end
# train_river epoch times CPU-only vs GPU-pipelined.
#
# Prerequisites on the rental box:
#   - repo cloned, HandRanks.dat present (repo root or Game/Utility/)
#   - python torch with CUDA >= 12.8 on RTX 50-series / Blackwell — the
#     wheel bundles libnvrtc used by the fused runtime-compiled solver
#   - built:  cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release \
#               -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')
#             cmake --build cmake-build-release --target rebel_hunl -j
#
# Run:  bash tools/rebel_rental_probe.sh
# Send back: rental_probe.log   (~20 min total)
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./cmake-build-release/rebel_hunl
OUT=rental_probe.log
NP=$(nproc)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
EXPECTED_GPUS=${EXPECTED_GPUS:-4}
MIN_MEM_GIB=${MIN_MEM_GIB:-230}
MIN_DISK_GB=${MIN_DISK_GB:-600}

{
echo "==================== box ===================="
date
echo "nproc: $NP"
free -g | head -2
df -h . | tail -1
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm \
    --format=csv 2>/dev/null
mem_gib=$(awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo)
disk_gb=$(df -PB1 . | awk 'NR==2{print int($4/1000000000)}')
[ "$NGPU" -eq "$EXPECTED_GPUS" ] \
    || { echo "FATAL: expected $EXPECTED_GPUS GPUs, found $NGPU"; exit 1; }
[ "$mem_gib" -ge "$MIN_MEM_GIB" ] \
    || { echo "FATAL: need >=${MIN_MEM_GIB} GiB RAM, found ${mem_gib}"; exit 1; }
[ "$disk_gb" -ge "$MIN_DISK_GB" ] \
    || { echo "FATAL: need >=${MIN_DISK_GB} GB free, found ${disk_gb}"; exit 1; }
python3 -c 'import sys,torch; expected=int(sys.argv[1]); cv=tuple(map(int,(torch.version.cuda or "0.0").split(".")[:2])); caps=[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]; arch=torch.cuda.get_arch_list(); print("torch",torch.__version__,"cuda",torch.version.cuda,"arch",arch); [print(i,torch.cuda.get_device_name(i),caps[i]) for i in range(len(caps))]; sys.exit("CUDA is unavailable") if not torch.cuda.is_available() else None; sys.exit(f"expected {expected} CUDA devices, found {len(caps)}") if len(caps)!=expected else None; sys.exit("rental GPUs are not all Blackwell sm_120") if any(c < (12,0) for c in caps) else None; sys.exit("Blackwell requires a CUDA >=12.8 torch wheel") if cv < (12,8) else None; sys.exit("torch wheel lacks sm_120 kernels") if "sm_120" not in arch else None' "$EXPECTED_GPUS"

echo "==================== validation ===================="
# kernels: CPU fast-vs-brute; gpu_check: eager + fused NVRTC kernel vs
# fp64 CPU solver (first Blackwell run of the runtime-compiled kernel);
# gadget_check: safe-resolving guarantee. The expanded kernel line also
# proves this is the rebuilt binary, not a stale executable beside new scripts.
kernel_line=$($BIN kernels | tail -1)
echo "$kernel_line"
echo "$kernel_line" | grep -Eq \
    'root-weight err .*topology-key=exact +PASS$' \
    || { echo "FATAL: kernels failed or rebel_hunl binary is stale"; exit 1; }
$BIN gpu_check 16 200
REBEL_GPU_F64=1 $BIN gpu_check 16 200
$BIN gadget_check 400

if [ -s "${ORACLE:-cand_f1.pt}" ]; then
    echo "==================== turn solver validation/bench ===================="
    for G in $(seq 0 $(( NGPU - 1 ))); do
        echo "--- GPU $G fused sanity:"
        CUDA_VISIBLE_DEVICES=$G $BIN gpu_check 4 60
        if [ "$G" -eq 0 ]; then
            turn_out=$(CUDA_VISIBLE_DEVICES=$G \
                REBEL_CKPT="${ORACLE:-cand_f1.pt}" REBEL_PCFR=0 \
                REBEL_TF32=1 $BIN turn_gpu_check 4 40)
            echo "$turn_out"
            [ "$(echo "$turn_out" | tail -1)" = "turn_gpu_check: PASS" ] \
                || { echo "FATAL: CFR+ turn check failed/stale"; exit 1; }
        fi
        turn_out=$(CUDA_VISIBLE_DEVICES=$G \
            REBEL_CKPT="${ORACLE:-cand_f1.pt}" REBEL_PCFR=1 \
            REBEL_TF32=1 $BIN turn_gpu_check 4 40)
        echo "$turn_out"
        [ "$(echo "$turn_out" | tail -1)" = "turn_gpu_check: PASS" ] \
            || { echo "FATAL: PCFR+ turn check failed/stale"; exit 1; }
    done
    # B=1 answers whether fused solving helps interactive play; the larger
    # batches select the generation setting. Do not infer one from the other.
    for B in 1 32 64 96 128 160 192 256; do
        REBEL_CKPT="${ORACLE:-cand_f1.pt}" REBEL_PCFR=1 REBEL_TF32=1 \
            REBEL_CPU_REF=0 \
            $BIN turn_gpu_bench "$B" 120 || \
            echo "turn_gpu_bench B=$B failed (VRAM?)"
    done

    echo "==================== PCFR efficacy recheck ===================="
    # Re-run one fixed river/turn comparison with the current featurizer and
    # oracle. PCFR+q is expected to lose on exact rivers (where it stays off)
    # and beat CFR+ on the net-leaf turn cases used by generation/play.
    CUDA_VISIBLE_DEVICES=0 REBEL_CKPT="${ORACLE:-cand_f1.pt}" \
        REBEL_TF32=1 $BIN pcfr_bench 1

    echo "==================== end-to-end train_turn ===================="
    echo "--- pure fused-GPU lane (1 epoch x 1024 rows):"
    REBEL_CKPT="${ORACLE:-cand_f1.pt}" REBEL_TF32=1 REBEL_PCFR=1 \
        REBEL_RIVER_ROWS=0 REBEL_HARVEST=0 REBEL_SGD_STEPS=0 \
        REBEL_PROBE_K=0 REBEL_THREADS=12 REBEL_GPU_TURN_BATCH=128 \
        REBEL_GPU_FRAC=1.0 $BIN train_turn 1 1024 2>&1 | \
        grep -E 'HUNL|epoch'
    echo "--- pure CPU lane (1 epoch x 256 rows):"
    REBEL_CKPT="${ORACLE:-cand_f1.pt}" REBEL_TF32=1 REBEL_PCFR=1 \
        REBEL_RIVER_ROWS=0 REBEL_HARVEST=0 REBEL_SGD_STEPS=0 \
        REBEL_PROBE_K=0 REBEL_THREADS="$NP" REBEL_GPU_TURN_BATCH=0 \
        $BIN train_turn 1 256 2>&1 | grep -E 'HUNL|epoch'
else
    echo "turn solver validation skipped: ${ORACLE:-cand_f1.pt} not found"
fi

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
