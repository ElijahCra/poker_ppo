#!/usr/bin/env bash
# tune_speed.sh — find the fastest training shape on THIS machine's GPU by
# rebuilding at each candidate config and timing the real training loop,
# then printing a ranked samples/sec table. Restores your original config +
# binary on exit (even if interrupted).
#
# Run it from the repo root in the shell that builds the project:
#     tools/tune_speed.sh
#     tools/tune_speed.sh "512:96:16 1024:48:16 1024:48:8"   # custom grid
#
# ── what it tunes ───────────────────────────────────────────────────────────
# Three compile-time knobs in config::kPPOConfig set the speed shape:
#   num_envs       parallel envs   → rollout inference batch (launch-bound)
#   num_steps      steps/rollout   → with num_envs sets batch_size
#   num_minibatches→ minibatch_size = num_envs*num_steps/num_minibatches
#                                    → update GEMM efficiency
# They split into two INDEPENDENT speed axes:
#   A) ROLLOUT SHAPE — num_envs × num_steps at FIXED batch_size & minibatch.
#      Dynamics-neutral (batch/minibatch/grad-steps all unchanged; only the
#      GAE truncation horizon shifts, negligible at gae_lambda≈0.9). Adopt
#      the fastest freely.
#   B) MINIBATCH SIZE — vary num_minibatches at a fixed rollout shape.
#      Bigger minibatch = better GPU utilisation BUT fewer gradient steps
#      per update = a real OPTIMISATION-DYNAMICS change. This script only
#      measures its SPEED; the fastest must be confirmed not to raise
#      exploitability before adopting:
#          POKER_PPO_BR_SEEDS=3 POKER_PPO_MAX_STEPS=37000000 ./poker_ppo
#      against the current default (paired — seeds are pinned in config).
#
# Two-stage workflow (saves rebuilds): first run tools/bench_throughput.sh
# (no rebuild) to find the rollout-throughput knee and pick 2-3 num_envs
# candidates, THEN list those here. The rollout benchmark can't see the
# update (the dominant, compute-bound half), which is why this script times
# the real loop end-to-end.
#
# Safety: edits include/config.h in place but restores it on EXIT. If the
# machine dies mid-run, recover with:  git checkout include/config.h
set -uo pipefail
cd "$(dirname "$0")/.."

CFG=include/config.h
BIN=cmake-build-release/poker_ppo
BUILD_DIR=cmake-build-release
STEPS_CAP=2000000          # ~40 updates: enough to reach steady full-clock
RUN_TIMEOUT=240            # seconds per point (raise for very large batches)

# Grid of "num_envs:num_steps:num_minibatches". Default: rollout-shape axis
# at fixed batch=49152/minibatch=3072 (first rows), then the minibatch axis
# at one rollout shape. Override via $1 (space-separated) or edit here.
if [ $# -ge 1 ]; then
  read -r -a GRID <<<"$1"
else
  GRID=(
    "384:128:16"   # batch 49152  mb 3072   ── rollout-shape axis (safe)
    "768:64:16"    # batch 49152  mb 3072
    "1024:48:16"   # batch 49152  mb 3072
    "1536:32:16"   # batch 49152  mb 3072
    "768:64:8"     # batch 49152  mb 6144   ── minibatch axis (BR-validate!)
    "768:64:4"     # batch 49152  mb 12288
    "768:64:2"     # batch 49152  mb 24576
  )
fi

if [ ! -d "$BUILD_DIR" ]; then echo "no $BUILD_DIR (configure cmake first)" >&2; exit 1; fi

cp "$CFG" "$CFG.tune.bak"
restore() {
  if [ -f "$CFG.tune.bak" ]; then
    mv -f "$CFG.tune.bak" "$CFG"
    echo >&2; echo "[tune] restoring original config + rebuilding..." >&2
    cmake --build "$BUILD_DIR" -j >/dev/null 2>&1 || true
  fi
}
trap restore EXIT

# Replace the three knobs ONLY inside the kPPOConfig block (the kBRConfig
# block below it has the same field names — the range address protects it).
set_cfg() {
  sed -i "/static constexpr PPOConfig kPPOConfig/,/^};/{
    s/\(\.num_envs[[:space:]]*=[[:space:]]*\)[0-9][0-9]*/\1$1/
    s/\(\.num_steps[[:space:]]*=[[:space:]]*\)[0-9][0-9]*/\1$2/
    s/\(\.num_minibatches[[:space:]]*=[[:space:]]*\)[0-9][0-9]*/\1$3/
  }" "$CFG"
}

# Median of a `rollout=`/`update=` field over steady updates (skips update 0,
# which includes one-time CUDA-graph capture).
med_ms() {
  grep -E '^\[update [1-9]' | sed -n "s/.*$1=\([0-9.][0-9.]*\)ms.*/\1/p" \
    | sort -n | awk '{a[NR]=$1} END{print (NR ? a[int((NR+1)/2)] : 0)}'
}

printf '%-16s %9s %9s %9s %12s %9s\n' \
       "envs:steps:mb" batch roll_ms upd_ms "samples/s" VRAM_MB
printf '%-16s %9s %9s %9s %12s %9s\n' \
       "-------------" ----- ------- ------ --------- -------
declare -a NAMES SPS
for pt in "${GRID[@]}"; do
  IFS=: read -r E S M <<<"$pt"
  echo "[tune] building $pt ..." >&2
  set_cfg "$E" "$S" "$M"
  if ! cmake --build "$BUILD_DIR" -j >/dev/null 2>&1; then
    printf '%-16s %9s\n' "$pt" "BUILD_FAIL"; continue
  fi
  echo "[tune] timing  $pt ..." >&2
  POKER_PPO_MAX_STEPS=$STEPS_CAP timeout $RUN_TIMEOUT "$BIN" >/tmp/tune.out 2>/dev/null &
  RUN=$!
  vram=0
  while kill -0 $RUN 2>/dev/null; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "${u:-}" ] && [ "$u" -gt "$vram" ] 2>/dev/null; then vram=$u; fi
    sleep 2
  done
  wait $RUN 2>/dev/null || true
  roll=$(med_ms rollout </tmp/tune.out)
  upd=$(med_ms update  </tmp/tune.out)
  batch=$((E * S))
  sps=$(awk "BEGIN{t=$roll+$upd; print (t>0 ? $batch/t*1000 : 0)}")
  printf '%-16s %9d %9s %9s %12.0f %9s\n' "$pt" "$batch" "$roll" "$upd" "$sps" "$vram"
  NAMES+=("$pt"); SPS+=("$sps")
done

# Winner.
best=-1; bestn=""
for i in "${!SPS[@]}"; do
  if awk "BEGIN{exit !(${SPS[$i]} > $best)}"; then best=${SPS[$i]}; bestn=${NAMES[$i]}; fi
done
echo
echo "Fastest: $bestn  (${best%.*} samples/s)"
cat <<'NOTE'

Adopt by editing config::kPPOConfig (num_envs / num_steps / num_minibatches).
  • If the winner only changed the ROLLOUT SHAPE (same batch_size & minibatch
    as your current default), it's dynamics-neutral — adopt directly.
  • If it changed MINIBATCH SIZE (or batch_size), run a 3-seed BR A/B first:
      POKER_PPO_BR_SEEDS=3 POKER_PPO_MAX_STEPS=37000000 ./cmake-build-release/poker_ppo
    and compare its bb/hand bound to the current default before committing.
Config + binary have been restored to your original.
NOTE
