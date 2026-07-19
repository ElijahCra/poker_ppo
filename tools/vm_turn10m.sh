#!/usr/bin/env bash
# 10M-turn-row generation campaign (DeepStack-scale street-2 data).
#
#   cd /poker_ppo/cmake-build-release && bash ../tools/vm_turn10m.sh
#
# Generates TARGET_ROWS street-2 (turn-root) rows into OUT using every
# lane at once: the GPU batch solver (fused kernel + on-device
# featurization, PCFR+) and the CPU worker pool (river emission OFF —
# river data is saturated; every episode yields exactly one 40KB
# street-2 row). Chunked + resumable: rerun after any death and it
# continues from the on-disk row count. The champion ckpt prices the
# net leaves (copied first — the trainer saves its ckpt path per epoch,
# and the champion must never be written in place).
#
# Tune GPU_FRAC once per box: run a short chunk, read the epoch wall
# with the lane split, and set it to gpu_rate/(gpu_rate+cpu_rate).
# Defaults assume a 5090 + ~380 vCPUs (GPU ~0.3 of throughput).
set -uo pipefail

BIN=./rebel_hunl
ORACLE=${ORACLE:-cand_f1.pt}          # champion of record prices leaves
TARGET_ROWS=${TARGET_ROWS:-10000000}
OUT=${OUT:-turn10m.bin}
CHUNK_EPISODES=${CHUNK_EPISODES:-20000}   # episodes per epoch
EPOCHS_PER_RUN=${EPOCHS_PER_RUN:-25}      # one resumable chunk
GPU_FRAC=${GPU_FRAC:-0.3}
GPU_BATCH=${GPU_BATCH:-128}           # 5090: 128-192; 8GB cards: <=64
THREADS=${THREADS:-0}                 # 0 = (cores - 8) / instances
T_TURN=${T_TURN:-120}
SEED_BASE=${SEED_BASE:-0}
ROW=40008

rows() {
    local s
    s=$(stat -c%s "$1" 2>/dev/null) || { echo 0; return; }
    if [ "$s" -ge 16 ]; then echo $(( (s - 16) / ROW )); else echo 0; fi
}
die() { echo "FATAL: $*" >&2; exit 1; }

# ── multi-GPU launcher: one pinned instance per device ───────────────
# Each shard writes its own OUT file with its own seed base (chunk
# seeds MUST differ across shards or they generate identical
# situations). Merge at mix-build time: cp shard0 mix && for the rest
# tail -c +17 shard >> mix.
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NGPU" -ge 1 ] || NGPU=1
if [ "$NGPU" -gt 1 ] && [ -z "${TURN10M_CHILD:-}" ]; then
    [ -x "$BIN" ] || die "$BIN not found — run from cmake-build-release"
    [ -s "$ORACLE" ] || die "$ORACLE missing"
    pgrep -f 'rebel_hun[l]' >/dev/null \
        && die "rebel_hunl already running — kill strays first"
    cp -n "$ORACLE" gen_oracle.pt
    cores=$(nproc)
    # DEDICATED lanes (measured locally: mixing lanes on one GPU loses
    # ~2x — the fused kernel's persistent windows head-block the CPU
    # workers' little oracle forwards). GPUs 0..N-2 run PURE fused lanes
    # (support threads only); the LAST GPU hosts the pure CPU lane's
    # small forwards and no fused kernels. CPUSHARE = CPU lane's slice
    # of the row target.
    CPUSHARE=${CPUSHARE:-0.15}
    cpu_rows=$(awk -v t="$TARGET_ROWS" -v s="$CPUSHARE" \
                   'BEGIN{printf "%d", t*s}')
    gpu_rows=$(( (TARGET_ROWS - cpu_rows + NGPU - 2) / (NGPU - 1) ))
    tsup=12
    tcpu=$(( cores - tsup * (NGPU - 1) - 4 ))
    [ "$tcpu" -ge 8 ] || tcpu=8
    echo "== launcher: $((NGPU-1)) pure-GPU lanes x $gpu_rows rows" \
         "($tsup thr each) + CPU lane $cpu_rows rows ($tcpu thr)"
    pids=()
    for g in $(seq 0 $(( NGPU - 2 ))); do
        CUDA_VISIBLE_DEVICES=$g TURN10M_CHILD=1 NGPU=1 \
        TARGET_ROWS=$gpu_rows OUT="${OUT%.bin}_g$g.bin" THREADS=$tsup \
        SEED_BASE=$(( g * 997 )) GPU_FRAC=1.0 GPU_BATCH=$GPU_BATCH \
        bash "$0" > "launcher_g$g.log" 2>&1 &
        pids+=("$!")
    done
    g=$(( NGPU - 1 ))
    CUDA_VISIBLE_DEVICES=$g TURN10M_CHILD=1 NGPU=1 \
    TARGET_ROWS=$cpu_rows OUT="${OUT%.bin}_g$g.bin" THREADS=$tcpu \
    SEED_BASE=$(( g * 997 )) GPU_FRAC=0.0 GPU_BATCH=0 \
    bash "$0" > "launcher_g$g.log" 2>&1 &
    pids+=("$!")
    fail=0
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    echo "== GPU shards:"
    for g in $(seq 0 $(( NGPU - 1 ))); do
        echo "   ${OUT%.bin}_g$g.bin: $(rows "${OUT%.bin}_g$g.bin") rows"
    done
    [ "$fail" -eq 0 ] || die "a shard failed — see launcher_g*.log"
    echo "== merge when building the training mix:"
    echo "   cp ${OUT%.bin}_g0.bin mix.bin"
    echo "   for g in \$(seq 1 $((NGPU-1))); do tail -c +17 ${OUT%.bin}_g\$g.bin >> mix.bin; done"
    exit 0
fi
[ "$THREADS" -gt 0 ] 2>/dev/null || THREADS=$(( $(nproc) - 8 ))

# ── gates ─────────────────────────────────────────────────────────────
[ -x "$BIN" ] || die "$BIN not found — run from cmake-build-release"
[ -s "$ORACLE" ] || die "$ORACLE missing — the champion ckpt prices leaves"
if [ -z "${TURN10M_CHILD:-}" ]; then   # children run beside siblings
    pgrep -f 'rebel_hun[l]' >/dev/null \
        && die "rebel_hunl already running — kill strays first"
fi
have=$(rows "$OUT")
need_gb=$(( (TARGET_ROWS - have) * ROW / 1000000000 + 5 ))
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge "$need_gb" ] \
    || die "need ~${need_gb}GB free for $((TARGET_ROWS - have)) more rows, have ${free_gb}GB"
if [ "$have" -gt 0 ]; then
    [ $(( ($(stat -c%s "$OUT") - 16) % ROW )) -eq 0 ] \
        || die "$OUT misaligned (torn rows) — move it aside"
    echo "== resuming: $have/$TARGET_ROWS rows present"
fi
cp -n "$ORACLE" gen_oracle.pt   # never write the champion in place

# ── generation loop (chunked, resumable) ─────────────────────────────
while true; do
    have=$(rows "$OUT")
    [ "$have" -ge "$TARGET_ROWS" ] && break
    remaining=$(( TARGET_ROWS - have ))
    # ~1 row per episode; don't overshoot the last chunk
    eps=$CHUNK_EPISODES
    epochs=$EPOCHS_PER_RUN
    if [ $(( eps * epochs )) -gt "$remaining" ]; then
        epochs=$(( (remaining + eps - 1) / eps ))
    fi
    echo "== chunk: $epochs x $eps episodes  ($have/$TARGET_ROWS rows, $(date '+%H:%M'))"
    REBEL_TF32=1 REBEL_PCFR=1 REBEL_RIVER_ROWS=0 REBEL_HARVEST=0 \
    REBEL_SGD_STEPS=0 REBEL_PROBE_K=0 \
    REBEL_SEED=$(( 7000 + SEED_BASE + have % 100000 )) \
    REBEL_CKPT=gen_oracle.pt REBEL_DATA_OUT="$OUT" \
    REBEL_T_TURN_TRAIN=$T_TURN REBEL_THREADS=$THREADS \
    REBEL_GPU_TURN_BATCH=$GPU_BATCH REBEL_GPU_FRAC=$GPU_FRAC \
    $BIN train_turn "$epochs" "$eps" > "gen_chunk_$have.log" 2>&1 &
    pid=$!
    # done when the final epoch line appears (appends precede it) —
    # CUDA teardown at exit is known to hang, so kill instead of wait
    while true; do
        if grep -Eq "epoch +${epochs} " "gen_chunk_$have.log" 2>/dev/null; then
            sleep 20
            kill -9 "$pid" 2>/dev/null
            break
        fi
        kill -0 "$pid" 2>/dev/null || {
            grep -Eq "epoch +${epochs} " "gen_chunk_$have.log" 2>/dev/null && break
            echo "== chunk DIED — last lines:" >&2
            tail -5 "gen_chunk_$have.log" >&2
            die "generation chunk failed (see gen_chunk_$have.log)"
        }
        sleep 60
    done
done

echo "== DONE: $(rows "$OUT") rows in $OUT"
[ $(( ($(stat -c%s "$OUT") - 16) % ROW )) -eq 0 ] || die "$OUT misaligned"
cat <<'EOF'
== next: training phase (manual — RAM and seed count are budget calls) ==
# mix with river ballast (~10:1 turn:river is plenty now):
#   cp turn10m.bin mix10m.bin && tail -c +17 river_hs_vm.bin >> mix10m.bin
# single seed first (preflop solving buffers the net lottery):
#   cp cand_f1.pt g3_s1.pt
#   REBEL_TF32=1 REBEL_GPU_CACHE=60000 REBEL_SEED=301 REBEL_CKPT=g3_s1.pt \
#   REBEL_DATA_IN=mix10m.bin REBEL_REPLAY_CAP=6000000 REBEL_PROBE_K=0 \
#   REBEL_LR=2e-4 REBEL_LR_FINAL=5e-5 REBEL_SGD_STEPS=4000 REBEL_BATCH=2048 \
#   ./rebel_hunl train_turn 40 0
# gate vs the 1.5248 reference (paired seed 1234):
#   REBEL_PREFLOP=1 REBEL_T_TURN=240 REBEL_T_RIVER=800 REBEL_CKPT=g3_s1.pt \
#   REBEL_SEED=1234 REBEL_LBR_LOG=gate_g3.csv ./rebel_hunl lbr 20000 128
# gadget calibration arms (same net, one knob each):
#   REBEL_GADGET=1                    (exact alts, delta 0)
#   REBEL_GADGET=1 REBEL_GADGET_DELTA=0.05
EOF
