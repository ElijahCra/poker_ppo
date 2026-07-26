#!/usr/bin/env bash
# Smoke the ENTIRE gen-3 campaign at toy scale (~25-35 min, ~$1), then
# tear down the toy artifacts and launch the real vm_gen3.sh.
#
#   cd /poker_ppo/cmake-build-release && bash ../tools/vm_gen3_test.sh
#
# Exercises every phase with the same code paths and the SAME artifact
# names the real run uses: 2k turn rows across all GPUs, 1k river rows,
# ~160 flop rows, tiny mix, both trainings (incl. the 1536x6 spec net), and
# all model/final gates plus pressure and latency probes (bounds are
# meaningless noise at toy scale — mechanics
# only). On success every artifact is deleted and verified gone (a
# leftover toy mix.bin would make the real run skip generation and
# train on 4k rows), then the full campaign starts. On failure the
# artifacts stay for inspection and the real run does NOT start.
# Set REBEL_SMOKE_ONLY=1 to stop after a successful toy teardown instead of
# immediately entering the paid/full campaign.
set -uo pipefail
cd "$(dirname "$0")/../cmake-build-release" 2>/dev/null || true

SELF_DIR=$(cd "$(dirname "$0")" && pwd)

echo "==== static launcher checks ===="
bash -n "$SELF_DIR/vm_gen3.sh" || exit 1
bash -n "$SELF_DIR/vm_turn10m.sh" || exit 1
REBEL_CHUNK_PLAN_CHECK=1 TARGET_ROWS=1250000 \
    bash "$SELF_DIR/vm_gen3.sh" | grep -q 'plan PASS: 3 chunks, 1250000 rows' \
    || exit 1
REBEL_CHUNK_PLAN_CHECK=1 TARGET_ROWS=1416667 SEED_BASE=200000000 \
    bash "$SELF_DIR/vm_turn10m.sh" | grep -q 'plan PASS:' || exit 1

ARTIFACTS="mix.bin river_g3.bin flop_g3.bin turn_g0.bin turn_g1.bin \
turn_g2.bin turn_g3.bin g3_s1.pt g3_spec.pt g3_selected.pt g3_final.pt \
g3_selected.meta g3_final.meta root_refine.bin root_refine.log \
g3_latency.jsonl g3_latency.err gen_oracle.pt flop_oracle.pt \
river_g3.log flop_g3.log .vm_gen3_seed_scheme"

echo "==== GEN-3 TOY SMOKE ($(date '+%H:%M')) ===="
for f in $ARTIFACTS; do
    [ -e "$f" ] && {
        echo "FATAL: $f already exists — this must start clean (real-run"
        echo "state? move it aside or run vm_gen3.sh directly)" >&2
        exit 1
    }
done

TARGET_ROWS=2000 CHUNK_EPISODES=250 EPOCHS_PER_RUN=2 \
DISK_HEADROOM_GB=2 \
RIVER_TARGET=1000 RIVER_EPOCHS=2 RIVER_EPS=500 \
FLOP_EPOCHS=2 FLOP_EPS=20 FLOP_THREADS=20 \
TRAIN_EPOCHS=2 TRAIN_SGD=50 REPLAY_CAP=100000 \
GPU_CACHE=2000 \
GATE_HANDS=50 GATE_THREADS=16 \
GATE_MAX_REGRESSION=100 PRESSURE_HANDS=4 PRESSURE_THREADS=2 \
ROOT_EPOCHS=1 ROOT_EPISODES=1 ROOT_SGD=20 ROOT_THREADS=2 LATENCY_HANDS=2 \
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

# The production bug this smoke must guard: default 500k chunks used to
# reduce `have` modulo 100k, giving every chunk the same seed.  Child launch
# logs now print the absolute-row seed; require global uniqueness.
mapfile -t chunk_seeds < <(
    sed -n 's/.*== chunk:.*seed=\([0-9][0-9]*\).*/\1/p' \
        turn_launcher_g*.log 2>/dev/null
)
[ "${#chunk_seeds[@]}" -gt 0 ] || {
    echo "FATAL: no turn chunk seeds found in launcher logs" >&2
    exit 1
}
dupe_seeds=$(printf '%s\n' "${chunk_seeds[@]}" | sort | uniq -d)
[ -z "$dupe_seeds" ] || {
    echo "FATAL: duplicate turn chunk seeds: $dupe_seeds" >&2
    exit 1
}

mix_bytes=$(stat -c%s mix.bin)
mix_rows=$(( (mix_bytes - 16) / 40008 ))
min_mix_rows=$(( 2000 + 1000 + 2 * 20 * 3 ))
[ "$mix_rows" -ge "$min_mix_rows" ] || {
    echo "FATAL: toy mix has $mix_rows rows; expected at least $min_mix_rows" >&2
    exit 1
}
echo "==== seed/row checks passed: ${#chunk_seeds[@]} unique chunks, $mix_rows mix rows ===="

echo ""
echo "==== SMOKE PASSED — toy gate lines (mechanics only, n=50): ===="
grep -H 'LBR vs ReBeL agent:' gate_g3*.log 2>/dev/null

echo "==== tearing down toy artifacts ===="
rm -f $ARTIFACTS
rm -f turn_g*_chunk*.log turn_launcher_g*.log
rm -f train_g3_s1.log train_g3_spec.log
rm -f gate_g3*.log gate_g3*.csv*
for f in mix.bin river_g3.bin flop_g3.bin g3_s1.pt g3_spec.pt g3_final.pt \
         train_g3_s1.log gate_g3champ_cfr.log; do
    [ -e "$f" ] && {
        echo "FATAL: teardown left $f — real run NOT started" >&2
        exit 1
    }
done
if [ "${REBEL_SMOKE_ONLY:-0}" = 1 ]; then
    echo "==== smoke-only requested — real campaign NOT started ===="
    exit 0
fi
echo "==== clean — launching the REAL campaign ===="
exec bash "$SELF_DIR/vm_gen3.sh"
