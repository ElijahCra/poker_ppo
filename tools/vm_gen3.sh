#!/usr/bin/env bash
# Gen-3 campaign, end to end, one command:
#
#   cd /poker_ppo/cmake-build-release && bash ../tools/vm_gen3.sh
#
# Shaped for 4x5090 / 128 cores / 256GB RAM / 1TB disk. Phases, each
# gated on verifiable on-disk state (rerun after any death — finished
# phases are detected and skipped):
#   0  sanity trio (kernels, gpu_check, flopctx_check)
#   1  generation, three concurrent lanes:
#        turn  — TARGET_ROWS street-2 rows, one pinned instance per GPU
#                (fused kernel + device featurization + PCFR+)
#        river — RIVER_TARGET exact rows on GPU 0 (net-independent)
#        flop  — FLOP_EPOCHS x FLOP_EPS street-1/2 rows on the CPU pool
#   2  mix build (mv + append + delete shards: disk peak stays ~one copy)
#   3  training, sequential (256GB fits ONE 5M-row replay at a time):
#        g3_s1  — continued from the champion (1024x2)
#        g3_big — capacity probe, fresh 2048x3
#   4  gates, sequential, paired seed 1234 vs the 1.5248 reference:
#        g3_s1 preflop | g3_s1 preflop+gadget(exact alts) | g3_big preflop
set -uo pipefail

BIN=${BIN:-./rebel_hunl}
CHAMPION=${CHAMPION:-cand_f1.pt}
CHAMPION_SIZE=42470807
TARGET_ROWS=${TARGET_ROWS:-10000000}
RIVER_TARGET=${RIVER_TARGET:-1000000}
RIVER_EPOCHS=${RIVER_EPOCHS:-40}
RIVER_EPS=${RIVER_EPS:-25000}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-40}
TRAIN_SGD=${TRAIN_SGD:-4000}
FLOP_EPOCHS=${FLOP_EPOCHS:-25}
FLOP_EPS=${FLOP_EPS:-500}
FLOP_THREADS=${FLOP_THREADS:-40}
CHUNK_EPISODES=${CHUNK_EPISODES:-20000}
EPOCHS_PER_RUN=${EPOCHS_PER_RUN:-25}
GPU_FRAC=${GPU_FRAC:-0.3}
GPU_BATCH=${GPU_BATCH:-128}
T_TURN=${T_TURN:-120}
REPLAY_CAP=${REPLAY_CAP:-5000000}      # 256GB box: 5M x 40KB = 200GB
GATE_HANDS=${GATE_HANDS:-20000}
GATE_THREADS=${GATE_THREADS:-128}
SEED_BASE=${SEED_BASE:-0}
ROW=40008

rows() {
    local s
    s=$(stat -c%s "$1" 2>/dev/null) || { echo 0; return; }
    if [ "$s" -ge 16 ]; then echo $(( (s - 16) / ROW )); else echo 0; fi
}
aligned() {
    local s
    s=$(stat -c%s "$1" 2>/dev/null) || return 1
    [ $(( (s - 16) % ROW )) -eq 0 ]
}
die() { echo "FATAL: $*" >&2; exit 1; }

# a generator is done when its log shows the final epoch (appends
# precede the print); CUDA teardown hangs, so kill instead of waiting
finish() {  # <pid> <log> <final_epoch> <label>
    local pid=$1 log=$2 fin=$3 label=$4
    while true; do
        if grep -Eq "epoch +${fin} " "$log" 2>/dev/null; then
            sleep 20
            kill -9 "$pid" 2>/dev/null
            echo "== $label: complete"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            grep -Eq "epoch +${fin} " "$log" 2>/dev/null && return 0
            echo "== $label DIED — tail:" >&2
            tail -5 "$log" >&2
            return 1
        fi
        sleep 60
    done
}

# ── turn-generation child (one pinned GPU instance) ──────────────────
if [ -n "${GEN3_TURN_CHILD:-}" ]; then
    OUT=$1
    while true; do
        have=$(rows "$OUT")
        [ "$have" -ge "$TARGET_ROWS" ] && break
        remaining=$(( TARGET_ROWS - have ))
        eps=$CHUNK_EPISODES
        epochs=$EPOCHS_PER_RUN
        if [ $(( eps * epochs )) -gt "$remaining" ]; then
            epochs=$(( (remaining + eps - 1) / eps ))
        fi
        REBEL_TF32=1 REBEL_PCFR=1 REBEL_RIVER_ROWS=0 REBEL_HARVEST=0 \
        REBEL_SGD_STEPS=0 REBEL_PROBE_K=0 \
        REBEL_SEED=$(( 7000 + SEED_BASE + have % 100000 )) \
        REBEL_CKPT=gen_oracle.pt REBEL_DATA_OUT="$OUT" \
        REBEL_T_TURN_TRAIN=$T_TURN REBEL_THREADS=$TURN_THREADS \
        REBEL_GPU_TURN_BATCH=$GPU_BATCH REBEL_GPU_FRAC=$GPU_FRAC \
        $BIN train_turn "$epochs" "$eps" > "${OUT%.bin}_chunk$have.log" 2>&1 &
        finish "$!" "${OUT%.bin}_chunk$have.log" "$epochs" "turn ${OUT}" \
            || exit 1
    done
    exit 0
fi

# ── gates ─────────────────────────────────────────────────────────────
[ -x "$BIN" ] || die "$BIN not found — run from cmake-build-release"
[ "$(stat -c%s "$CHAMPION" 2>/dev/null || echo 0)" -eq "$CHAMPION_SIZE" ] \
    || die "$CHAMPION missing or wrong size (want $CHAMPION_SIZE)"
[ -f ../HandRanks.dat ] || [ -f HandRanks.dat ] \
    || [ -f ../Game/Utility/HandRanks.dat ] \
    || die "HandRanks.dat not found — scp it (untracked in git)"
pgrep -f 'rebel_hun[l]' >/dev/null \
    && die "rebel_hunl already running — kill strays first"
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NGPU" -ge 1 ] || NGPU=1
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
need_gb=$(( (TARGET_ROWS - $(rows mix.bin)) * ROW / 1000000000 + 80 ))
[ "$(rows mix.bin)" -gt 0 ] || [ "$free_gb" -ge "$need_gb" ] \
    || die "need ~${need_gb}GB free, have ${free_gb}GB"

# ── phase 0: sanity ──────────────────────────────────────────────────
echo "== phase 0: sanity"
$BIN kernels 2>&1 | tail -1 | grep -q PASS || die "kernels FAIL"
$BIN gpu_check 4 60 2>&1 | tail -1 | grep -q PASS || die "gpu_check FAIL"
$BIN flopctx_check 10 2>&1 | tail -1 | grep -q PASS || die "flopctx FAIL"
echo "   all PASS"

# ── phase 1: generation, three lanes ─────────────────────────────────
cp -n "$CHAMPION" gen_oracle.pt
cp -n "$CHAMPION" flop_oracle.pt
cores=$(nproc)
TURN_THREADS=$(( (cores - FLOP_THREADS - 16) / NGPU ))
[ "$TURN_THREADS" -ge 4 ] || TURN_THREADS=4
export TURN_THREADS

RIVER_PID=; FLOP_PID=
if [ "$(rows river_g3.bin)" -lt "$RIVER_TARGET" ]; then
    [ "$(rows river_g3.bin)" -eq 0 ] \
        || die "river_g3.bin partial — move aside (seed reuse duplicates)"
    echo "== river lane: $RIVER_TARGET rows on GPU 0"
    CUDA_VISIBLE_DEVICES=0 \
    REBEL_TF32=1 REBEL_SEED=71 REBEL_CKPT= REBEL_PROBE_K=0 \
    REBEL_SGD_STEPS=0 REBEL_GPU_BATCH=8192 REBEL_GPU_FRAC=0.95 \
    REBEL_THREADS=8 REBEL_DATA_OUT=river_g3.bin \
    $BIN train_river "$RIVER_EPOCHS" "$RIVER_EPS" > river_g3.log 2>&1 &
    RIVER_PID=$!
else
    echo "== river lane: done ($(rows river_g3.bin) rows)"
fi

FLOP_TARGET=$(( FLOP_EPOCHS * FLOP_EPS * 3 ))
if [ "$(rows flop_g3.bin)" -lt "$FLOP_TARGET" ]; then
    [ "$(rows flop_g3.bin)" -eq 0 ] \
        || die "flop_g3.bin partial — move aside"
    echo "== flop lane: ~$(( FLOP_EPOCHS * FLOP_EPS * 4 )) rows on CPUs"
    CUDA_VISIBLE_DEVICES=$(( NGPU > 1 ? 1 : 0 )) \
    REBEL_TF32=1 REBEL_PCFR=1 REBEL_SEED=101 REBEL_CKPT=flop_oracle.pt \
    REBEL_PROBE_K=0 REBEL_SGD_STEPS=0 REBEL_THREADS=$FLOP_THREADS \
    REBEL_DATA_OUT=flop_g3.bin \
    $BIN train_flop "$FLOP_EPOCHS" "$FLOP_EPS" > flop_g3.log 2>&1 &
    FLOP_PID=$!
    sleep 90
    grep -Fq "[ckpt] loaded flop_oracle.pt" flop_g3.log || {
        kill -9 "$FLOP_PID" 2>/dev/null
        die "flop lane did not load the oracle — rows would be garbage"
    }
else
    echo "== flop lane: done ($(rows flop_g3.bin) rows)"
fi

if [ "$(rows mix.bin)" -eq 0 ]; then
    per=$(( (TARGET_ROWS + NGPU - 1) / NGPU ))
    turn_done=1
    for g in $(seq 0 $(( NGPU - 1 ))); do
        [ "$(rows "turn_g$g.bin")" -lt "$per" ] && turn_done=0
    done
    if [ "$turn_done" -eq 0 ]; then
        echo "== turn lane: $NGPU GPUs x $per rows, $TURN_THREADS thr each"
        tpids=()
        for g in $(seq 0 $(( NGPU - 1 ))); do
            CUDA_VISIBLE_DEVICES=$g GEN3_TURN_CHILD=1 \
            TARGET_ROWS=$per SEED_BASE=$(( g * 997 )) \
            bash "$0" "turn_g$g.bin" > "turn_launcher_g$g.log" 2>&1 &
            tpids+=("$!")
        done
        tfail=0
        for p in "${tpids[@]}"; do wait "$p" || tfail=1; done
        [ "$tfail" -eq 0 ] || die "a turn shard failed (turn_launcher_g*.log)"
    fi
    echo "== turn lane: complete"
fi
[ -n "$RIVER_PID" ] && { finish "$RIVER_PID" river_g3.log "$RIVER_EPOCHS" \
    "river lane" || die "river lane failed"; }
[ -n "$FLOP_PID" ] && { finish "$FLOP_PID" flop_g3.log "$FLOP_EPOCHS" \
    "flop lane" || die "flop lane failed"; }

# ── phase 2: mix (mv + append + delete keeps disk peak ~one copy) ───
if [ "$(rows mix.bin)" -eq 0 ]; then
    for f in turn_g*.bin river_g3.bin flop_g3.bin; do
        [ -f "$f" ] && { aligned "$f" || die "$f misaligned"; }
    done
    echo "== building mix.bin"
    mv turn_g0.bin mix.bin
    for g in $(seq 1 $(( NGPU - 1 ))); do
        [ -f "turn_g$g.bin" ] || continue
        tail -c +17 "turn_g$g.bin" >> mix.bin && rm "turn_g$g.bin"
    done
    tail -c +17 river_g3.bin >> mix.bin
    tail -c +17 flop_g3.bin >> mix.bin
fi
aligned mix.bin || die "mix.bin misaligned"
echo "== mix: $(rows mix.bin) rows"

# ── phase 3: training, sequential (RAM fits one replay) ─────────────
train_one() {  # <ckpt> <seed> <extra-env...>
    local ck=$1 sd=$2
    shift 2
    grep -Eq "epoch +${TRAIN_EPOCHS} +replay=" "train_${ck%.pt}.log" \
        2>/dev/null && {
        echo "== $ck: already trained"
        return 0
    }
    echo "== training $ck"
    env "$@" REBEL_TF32=1 REBEL_GPU_CACHE=60000 REBEL_SEED="$sd" \
        REBEL_CKPT="$ck" REBEL_DATA_IN=mix.bin \
        REBEL_REPLAY_CAP=$REPLAY_CAP REBEL_PROBE_K=0 \
        REBEL_SGD_STEPS=$TRAIN_SGD REBEL_BATCH=2048 \
        $BIN train_turn "$TRAIN_EPOCHS" 0 > "train_${ck%.pt}.log" 2>&1
    grep -Eq "epoch +${TRAIN_EPOCHS} +replay=" "train_${ck%.pt}.log" \
        || die "$ck training did not finish (train_${ck%.pt}.log)"
    grep -m1 'replay=' "train_${ck%.pt}.log" | grep -q 'replay= *0 ' \
        && die "$ck trained on EMPTY replay"
    return 0
}
[ -f g3_s1.pt ] || cp "$CHAMPION" g3_s1.pt
train_one g3_s1.pt 301 REBEL_LR=2e-4 REBEL_LR_FINAL=5e-5
train_one g3_big.pt 302 REBEL_LR=1e-3 REBEL_LR_FINAL=1e-4 \
    REBEL_HIDDEN=2048 REBEL_LAYERS=3

# ── phase 4: gates, sequential, paired seed 1234 ────────────────────
gate_one() {  # <name> <ckpt> <extra-env...>
    local name=$1 ck=$2
    shift 2
    grep -q 'LBR vs ReBeL agent:' "gate_$name.log" 2>/dev/null && {
        echo "== gate $name: done"
        return 0
    }
    rm -f "gate_$name.csv" "gate_$name.csv".*
    echo "== gate $name ($GATE_HANDS hands @ $GATE_THREADS thr)"
    env "$@" REBEL_PREFLOP=1 REBEL_T_TURN=240 REBEL_T_RIVER=800 \
        REBEL_CKPT="$ck" REBEL_SEED=1234 REBEL_LBR_LOG="gate_$name.csv" \
        $BIN lbr "$GATE_HANDS" "$GATE_THREADS" > "gate_$name.log" 2>&1
    grep -q 'LBR vs ReBeL agent:' "gate_$name.log" \
        || die "gate $name did not finish"
}
gate_one g3s1_pre    g3_s1.pt
gate_one g3s1_gadget g3_s1.pt REBEL_GADGET=1
gate_one g3big_pre   g3_big.pt REBEL_HIDDEN=2048 REBEL_LAYERS=3

echo ""
echo "== GEN-3 RESULTS ========================================="
for n in g3s1_pre g3s1_gadget g3big_pre; do
    echo "  $n  $(grep 'LBR vs ReBeL agent:' "gate_$n.log" 2>/dev/null || echo INCOMPLETE)"
done
echo "reference: champion cand_f1 preflop = 1.5248 (paired seed 1234)"
