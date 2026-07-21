#!/usr/bin/env bash
# Smoke the ENTIRE gen-3 campaign at toy scale (~25-35 min, ~$1), then
# tear down the toy artifacts and launch the real vm_gen3.sh.
#
#   cd /poker_ppo/cmake-build-release && bash ../tools/vm_gen3_test.sh
#
# Exercises every phase with the same code paths and the SAME artifact
# names the real run uses: 2k turn rows across all GPUs, 1k river rows,
# ~160 flop rows, tiny mix, both trainings (incl. the 2048x3 net), and
# all three 50-hand gates (bounds are meaningless noise — mechanics
# only). On success every artifact is deleted and verified gone (a
# leftover toy mix.bin would make the real run skip generation and
# train on 4k rows), then the full campaign starts. On failure the
# artifacts stay for inspection and the real run does NOT start.
set -uo pipefail
cd "$(dirname "$0")/../cmake-build-release" 2>/dev/null || true

SELF_DIR=$(cd "$(dirname "$0")" && pwd)

ARTIFACTS="mix.bin river_g3.bin flop_g3.bin turn_g0.bin turn_g1.bin \
turn_g2.bin turn_g3.bin g3_s1.pt g3_big.pt gen_oracle.pt flop_oracle.pt \
river_g3.log flop_g3.log"

echo "==== GEN-3 TOY SMOKE ($(date '+%H:%M')) ===="
for f in $ARTIFACTS; do
    [ -e "$f" ] && {
        echo "FATAL: $f already exists — this must start clean (real-run"
        echo "state? move it aside or run vm_gen3.sh directly)" >&2
        exit 1
    }
done

TARGET_ROWS=2000 CHUNK_EPISODES=250 EPOCHS_PER_RUN=2 \
RIVER_TARGET=1000 RIVER_EPOCHS=2 RIVER_EPS=500 \
FLOP_EPOCHS=2 FLOP_EPS=20 FLOP_THREADS=20 \
TRAIN_EPOCHS=2 TRAIN_SGD=50 REPLAY_CAP=100000 \
GATE_HANDS=50 GATE_THREADS=16 \
bash "$SELF_DIR/vm_gen3.sh"
rc=$?

if [ "$rc" -ne 0 ]; then
    echo ""
    echo "==== SMOKE FAILED (exit $rc) — real campaign NOT started ===="
    echo "artifacts left in place; check the phase that died:"
    ls -la turn_launcher_g*.log river_g3.log flop_g3.log \
        train_g3_*.log gate_g3*.log 2>/dev/null
    exit "$rc"
fi

echo ""
echo "==== SMOKE PASSED — toy gate lines (mechanics only, n=50): ===="
grep -H 'LBR vs ReBeL agent:' gate_g3*.log 2>/dev/null

echo "==== tearing down toy artifacts ===="
rm -f $ARTIFACTS
rm -f turn_g*_chunk*.log turn_launcher_g*.log
rm -f train_g3_s1.log train_g3_big.log
rm -f gate_g3s1_pre.log gate_g3s1_gadget.log gate_g3big_pre.log
rm -f gate_g3s1_pre.csv* gate_g3s1_gadget.csv* gate_g3big_pre.csv*
for f in mix.bin river_g3.bin flop_g3.bin g3_s1.pt g3_big.pt \
         train_g3_s1.log gate_g3s1_pre.log; do
    [ -e "$f" ] && {
        echo "FATAL: teardown left $f — real run NOT started" >&2
        exit 1
    }
done
echo "==== clean — launching the REAL campaign ===="
exec bash "$SELF_DIR/vm_gen3.sh"
