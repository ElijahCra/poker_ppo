#!/usr/bin/env bash
# Gated ReBeL flop/preflop pipeline for a rental VM.
#
#   cd /poker_ppo/cmake-build-release && bash ../tools/vm_flop_pipeline.sh
#
# Run from the directory holding rebel_hunl + rebel_hs_turn_v2.pt.
# Every phase refuses to start unless its inputs verifiably exist, so
# re-running after any failure is safe: finished phases are skipped.
# Phase 1 runs the river lane (GPU) and flop lane (CPU) concurrently;
# phase 2 mixes the datasets and trains 3 seeds offline; the LBR gate
# commands are printed at the end (run those by hand — they're the
# rental-time budget decision).
set -uo pipefail

BIN=./rebel_hunl
V2=rebel_hs_turn_v2.pt
V2_SIZE=42471298
ROW=40008                       # bytes/row: kdim 6687 f32 + 2652 f32 + 2652 u8
RIVER_EPOCHS=40; RIVER_EPS=25000   # 1M river rows (~20 min on a 5090)
FLOP_EPOCHS=50;  FLOP_EPS=500      # ~100k flop/turn rows (~2h, CPU-bound)

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

# A generator is DONE when its log shows the final epoch line (the data
# append precedes that print). CUDA teardown at process exit is known to
# hang on the rental, so we kill once the line appears instead of waiting.
finish() {  # <pid> <log> <final_epoch> <label>
    local pid=$1 log=$2 fin=$3 label=$4
    while true; do
        if grep -Eq "epoch +${fin} " "$log" 2>/dev/null; then
            sleep 30            # margin for the epoch's ckpt save
            kill -9 "$pid" 2>/dev/null
            echo "== $label: complete"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            grep -Eq "epoch +${fin} " "$log" 2>/dev/null && return 0
            echo "== $label DIED early — last log lines:" >&2
            tail -5 "$log" >&2
            return 1
        fi
        sleep 60
    done
}

# ── gates ─────────────────────────────────────────────────────────────
[ -x "$BIN" ] || die "$BIN not found — run from cmake-build-release"
[ "$(stat -c%s $V2 2>/dev/null || echo 0)" -eq "$V2_SIZE" ] \
    || die "$V2 missing or truncated (need $V2_SIZE bytes) — re-upload it"
pgrep -f 'rebel_hun[l]' >/dev/null \
    && die "rebel_hunl already running — decide, then: pkill -9 -f 'rebel_hun[l]'"
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 100 ] || die "need ~100GB free disk here, have ${free_gb}GB"

# ── phase 1: generation ──────────────────────────────────────────────
RIVER_TGT=$(( RIVER_EPOCHS * RIVER_EPS ))
FLOP_TGT_MIN=$(( FLOP_EPOCHS * FLOP_EPS * 3 ))   # ≥3 rows/episode floor
RIVER_PID=; FLOP_PID=

r=$(rows river_hs_vm.bin)
if [ "$r" -ge "$RIVER_TGT" ]; then
    echo "== river lane: $r rows present, skipping"
elif [ "$r" -gt 0 ]; then
    die "river_hs_vm.bin is PARTIAL ($r/$RIVER_TGT rows) — a rerun would \
append duplicate situations (same seed). Move it aside first."
else
    echo "== river lane: generating $RIVER_TGT rows (GPU lane, ~20 min)"
    REBEL_SEED=71 REBEL_CKPT= REBEL_PROBE_K=0 REBEL_SGD_STEPS=0 \
    REBEL_GPU_BATCH=8192 REBEL_GPU_FRAC=0.9 REBEL_THREADS=32 \
    REBEL_DATA_OUT=river_hs_vm.bin \
    $BIN train_river $RIVER_EPOCHS $RIVER_EPS > river_gen.log 2>&1 &
    RIVER_PID=$!
fi

f=$(rows flop_hs.bin)
if [ "$f" -ge "$FLOP_TGT_MIN" ]; then
    echo "== flop lane: $f rows present, skipping"
elif [ "$f" -gt 0 ]; then
    die "flop_hs.bin is PARTIAL ($f rows) — move it aside first."
else
    echo "== flop lane: generating ~$(( FLOP_EPOCHS * FLOP_EPS * 4 )) rows (CPU lane, ~2h)"
    cp "$V2" flop_s1.pt
    REBEL_SEED=101 REBEL_CKPT=flop_s1.pt REBEL_PROBE_K=0 REBEL_THREADS=340 \
    REBEL_DATA_OUT=flop_hs.bin \
    $BIN train_flop $FLOP_EPOCHS $FLOP_EPS > flop_gen.log 2>&1 &
    FLOP_PID=$!
    sleep 90
    # flop targets are bootstrapped through the net — a run that silently
    # started with random weights would write 100k rows of garbage
    grep -Fq "[ckpt] loaded flop_s1.pt" flop_gen.log || {
        kill -9 "$FLOP_PID" 2>/dev/null
        die "flop lane did NOT load flop_s1.pt (see flop_gen.log)"
    }
fi

[ -n "$FLOP_PID" ]  && { finish "$FLOP_PID"  flop_gen.log  "$FLOP_EPOCHS"  "flop lane"  || die "flop generation failed"; }
[ -n "$RIVER_PID" ] && { finish "$RIVER_PID" river_gen.log "$RIVER_EPOCHS" "river lane" || die "river generation failed"; }

aligned river_hs_vm.bin || die "river_hs_vm.bin misaligned (torn rows)"
aligned flop_hs.bin     || die "flop_hs.bin misaligned (torn rows)"
echo "== data ready: river=$(rows river_hs_vm.bin) flop=$(rows flop_hs.bin) rows"

# ── phase 2: mix + 3-seed offline training ──────────────────────────
want=$(( $(rows river_hs_vm.bin) + $(rows flop_hs.bin) ))
if [ "$(rows mix.bin)" -ne "$want" ]; then
    echo "== building mix.bin ($want rows)"
    rm -f mix.bin
    cp flop_hs.bin mix.bin
    tail -c +17 river_hs_vm.bin >> mix.bin
fi
aligned mix.bin || die "mix.bin misaligned"
echo "== mix: $(rows mix.bin) rows"

# SEQUENTIAL: three concurrent ~43GB dataset loads OOM-killed a seed on
# a rental (RAM varies by box); training is ~10 min/seed anyway. Each
# seed must show its final epoch WITH a non-empty replay to count —
# a missing log line means it died (OOM leaves no trace in its own log).
SEED_EPOCHS=30
for s in 1 2 3; do
    if grep -Eq "epoch +${SEED_EPOCHS} +replay=" seed$s.log 2>/dev/null; then
        echo "== seed $s: already trained, skipping"
        continue
    fi
    cp "$V2" cand_f$s.pt
    echo "== seed $s training (log seed$s.log)"
    REBEL_SEED=$((200+s)) REBEL_CKPT=cand_f$s.pt REBEL_DATA_IN=mix.bin \
    REBEL_REPLAY_CAP=1200000 REBEL_PROBE_K=0 \
    REBEL_LR=2e-4 REBEL_LR_FINAL=5e-5 REBEL_SGD_STEPS=2000 REBEL_BATCH=512 \
    $BIN train_flop $SEED_EPOCHS 0 > seed$s.log 2>&1
    grep -Eq "epoch +${SEED_EPOCHS} +replay=" seed$s.log || {
        tail -3 seed$s.log >&2
        die "seed $s did not finish (OOM/crash?) — cand_f$s.pt is NOT trained"
    }
    grep -m1 'replay=' seed$s.log | grep -q 'replay= *0 ' \
        && die "seed $s trained on an EMPTY replay — check seed$s.log"
done
echo "== training done: cand_f1.pt cand_f2.pt cand_f3.pt"

cat <<'EOF'
== next: LBR gates (CPU-bound, ~30 min each at 128 threads) ==
per candidate, preflop-solving arm + control (the 1.796 config of record):

REBEL_PREFLOP=1 REBEL_T_TURN=240 REBEL_T_RIVER=800 REBEL_CKPT=cand_f1.pt \
REBEL_SEED=1234 REBEL_LBR_LOG=gate_f1_pre.csv ./rebel_hunl lbr 20000 128

REBEL_PREFLOP=0 REBEL_BLUEPRINT=poker_ppo_model_nlhe_full_52.pt \
REBEL_T_TURN=240 REBEL_T_RIVER=800 REBEL_CKPT=cand_f1.pt \
REBEL_SEED=1234 REBEL_LBR_LOG=gate_f1_ctl.csv ./rebel_hunl lbr 20000 128
EOF
