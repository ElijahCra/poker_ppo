#!/usr/bin/env bash
# scale_config.sh — scale the training BATCH up by a factor S for a bigger
# GPU (H100/H200), preserving the invariants that keep self-play stable.
#
# A larger GPU has spare compute/VRAM. The throughput-and-stability way to
# spend it is a larger batch — NOT a faster small-batch loop (which leaves
# the GPU idle) and NOT an uncoordinated num_envs bump (which silently
# changes training dynamics). This scales the four coupled knobs together:
#
#   num_envs        *= S    batch grows S× (num_steps held → GAE horizon same)
#   num_minibatches *= S    holds minibatch_size → per-step gradient quality
#                           AND the captured update-graph shape unchanged
#   total_timesteps *= S    holds num_updates = total/batch = the number of
#                           self-play POLICY ITERATIONS (the quantity R-NaD /
#                           MMD last-iterate convergence actually depends on)
#   learning_rate   *= √S   gradient-noise scaling rule: an S× bigger batch
#                           has ~√S lower gradient noise → a √S larger step
#                           is the matched, stable choice
#
# Net: identical policy-iteration count and per-step gradient, with S× more
# on-policy data per update — lower variance, which is STABILISING. The only
# thing that changes is variance, so this is the dynamics-coherent scale.
# Still confirm with a best-response A/B before trusting a new S (large-batch
# returns diminish, and √S LR is a heuristic):
#   POKER_PPO_BR_SEEDS=3 POKER_PPO_MAX_STEPS=37000000 ./cmake-build-release/poker_ppo
#
# This does NOT change minibatch_size. Growing the minibatch (to fill an
# H100's SMs with bigger GEMMs) is a SEPARATE dynamics change — sweep it with
# tune_speed.sh and BR-validate it on its own.
#
# Usage:
#   tools/scale_config.sh <factor>            # dry-run: print the new shape
#   tools/scale_config.sh <factor> --apply    # edit config.h + rebuild
set -uo pipefail
cd "$(dirname "$0")/.."

CFG=include/config.h
BUILD_DIR="${BUILD_DIR:-cmake-build-release}"
S="${1:-}"
APPLY=0; [ "${2:-}" = "--apply" ] && APPLY=1
case "$S" in (''|*[!0-9]*) echo "usage: scale_config.sh <integer factor> [--apply]" >&2; exit 2;; esac
[ "$S" -ge 1 ] || { echo "factor must be >= 1" >&2; exit 2; }

read_i() {  # integer kPPOConfig field (kBRConfig protected by the range)
  sed -n "/static constexpr PPOConfig kPPOConfig/,/^};/{s/.*\.$1[[:space:]]*=[[:space:]]*\([0-9][0-9']*\).*/\1/p;}" "$CFG" \
    | head -1 | tr -d "'"
}
read_lr() {
  sed -n "/static constexpr PPOConfig kPPOConfig/,/^};/{s/.*\.learning_rate[[:space:]]*=[[:space:]]*\([0-9.eE+-]*\)f.*/\1/p;}" "$CFG" | head -1
}

envs=$(read_i num_envs); steps=$(read_i num_steps); mb=$(read_i num_minibatches)
tt=$(read_i total_timesteps); lr=$(read_lr)
[ -n "$envs" ] && [ -n "$mb" ] && [ -n "$tt" ] && [ -n "$lr" ] || {
    echo "could not parse kPPOConfig (run from repo root)" >&2; exit 1; }

new_envs=$(( envs * S )); new_mb=$(( mb * S )); new_tt=$(( tt * S ))
new_lr=$(awk "BEGIN{printf \"%.3e\", $lr*sqrt($S)}")    # e.g. 6.000e-04
new_lr="${new_lr}f"
batch=$(( envs * steps )); new_batch=$(( new_envs * steps ))
mbsize=$(( batch / mb ))

cat <<EOF
scale factor S=$S  (batch ${batch} → ${new_batch}, minibatch ${mbsize} held)
                       current        ->  scaled
  num_envs            $envs           ->  $new_envs
  num_steps           $steps           ->  $steps           (unchanged)
  num_minibatches     $mb           ->  $new_mb
  total_timesteps     $tt   ->  $new_tt
  learning_rate       ${lr}f        ->  $new_lr   (×√$S)
  --> num_updates (policy iterations) and minibatch_size: UNCHANGED
EOF

if [ "$APPLY" -ne 1 ]; then
    echo
    echo "dry-run. re-run with --apply to edit $CFG and rebuild."
    echo "then BR-validate: POKER_PPO_BR_SEEDS=3 POKER_PPO_MAX_STEPS=37000000 ./$BUILD_DIR/poker_ppo"
    exit 0
fi

cp "$CFG" "$CFG.scale.bak"
sed -i "/static constexpr PPOConfig kPPOConfig/,/^};/{
  s/\(\.num_envs[[:space:]]*=[[:space:]]*\)[0-9][0-9']*/\1$new_envs/
  s/\(\.num_minibatches[[:space:]]*=[[:space:]]*\)[0-9][0-9']*/\1$new_mb/
  s/\(\.total_timesteps[[:space:]]*=[[:space:]]*\)[0-9][0-9']*/\1$new_tt/
  s/\(\.learning_rate[[:space:]]*=[[:space:]]*\)[0-9.eE+-]*f/\1$new_lr/
}" "$CFG"
echo "edited $CFG (backup: $CFG.scale.bak); rebuilding..."
if cmake --build "$BUILD_DIR" -j --target poker_ppo; then
    echo "done. shape: num_envs=$new_envs num_steps=$steps num_minibatches=$new_mb"
    echo "      total_timesteps=$new_tt learning_rate=$new_lr"
    echo "NEXT: BR-validate before a full run —"
    echo "  POKER_PPO_BR_SEEDS=3 POKER_PPO_MAX_STEPS=37000000 ./$BUILD_DIR/poker_ppo"
    echo "  (revert: mv $CFG.scale.bak $CFG && cmake --build $BUILD_DIR -j)"
else
    echo "build failed — reverting config" >&2
    mv -f "$CFG.scale.bak" "$CFG"
    exit 1
fi
