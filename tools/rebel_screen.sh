#!/usr/bin/env bash
# ReBeL river-net architecture screen (2026-07-06).
#
# Waits for the generation run's river_data.bin to reach ROWS rows, then
# SNAPSHOTS it — every arm trains on byte-identical data (the generator
# keeps appending to the live file; comparing arms against a moving
# dataset would confound arm order with data volume). Arms run offline
# (episodes=0): pure SGD epochs, judged on the fixed row%50 heldout split
# and the printed target-var baseline. Probes are skipped (REBEL_PROBE_K=0)
# — probe the winner separately.
#
# Decision rule: if the big arms move heldout <~15% vs base, the river net
# is FEATURE-bound -> next step is hand-strength features, not width.
set -u
cd "$(dirname "$0")/.."

DATA=river_data.bin
SNAP=river_data_screen.bin
ROWS=300000            # ~75 generation epochs
ROWBYTES=24096         # feat f32[2709] + target f32[2652] + mask u8[2652]
NEED=$((16 + ROWS * ROWBYTES))

echo "[screen] waiting for $DATA to reach $ROWS rows ($NEED bytes)"
while [ ! -f "$DATA" ] || [ "$(stat -c%s "$DATA")" -lt "$NEED" ]; do
    sleep 120
done
echo "[screen] snapshotting to $SNAP"
cp "$DATA" "$SNAP"

run_arm() {
    local name=$1; shift
    echo "=== arm $name ($(date +%H:%M:%S)) ==="
    env REBEL_DATA_IN=$SNAP REBEL_PROBE_K=0 REBEL_SEED=1234 \
        REBEL_CKPT=screen_$name.pt "$@" \
        ./cmake-build-release/rebel_hunl train_river 30 0
}

# base = current arch on the new data distribution (the comparison anchor)
run_arm base       REBEL_SGD_STEPS=2000 REBEL_BATCH=512 REBEL_LR=1e-3 REBEL_LR_FINAL=1e-4
run_arm h2048      REBEL_HIDDEN=2048 REBEL_SGD_STEPS=2000 REBEL_BATCH=512 REBEL_LR=1e-3 REBEL_LR_FINAL=1e-4
run_arm h1024x3    REBEL_LAYERS=3 REBEL_SGD_STEPS=2000 REBEL_BATCH=512 REBEL_LR=1e-3 REBEL_LR_FINAL=1e-4
# ReBeL paper spec: 6x1536 GeLU+LayerNorm, batch 1024, Adam 3e-4
run_arm rebel_spec REBEL_GELU_LN=1 REBEL_HIDDEN=1536 REBEL_LAYERS=6 REBEL_SGD_STEPS=2000 REBEL_BATCH=1024 REBEL_LR=3e-4 REBEL_LR_FINAL=3e-5

echo "=== screen done ($(date +%H:%M:%S)) ==="
echo "compare: grep 'epoch  30' rebel_screen.log  (final heldout per arm)"
