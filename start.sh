#!/usr/bin/env bash
#
# Auto-resume or fresh-start launcher for poker_ppo.
#
# Logic:
#   1. Find the most recent run dir under runs/ (ISO timestamp prefix).
#   2. If it contains a `FINISHED` marker -> last run completed normally,
#      start a fresh run.
#   3. Else if it has at least one checkpoint -> resume that run.
#   4. Else (no FINISHED, no checkpoint) -> last run died before saving;
#      start fresh and leave the corpse for inspection.
#
# Usage:
#   ./start.sh                         # auto-detect
#   ./start.sh --force-fresh           # ignore any in-progress run
#   ./start.sh --force-resume <dir>    # resume a specific run dir
#   ./start.sh -- --strategy serial    # extra args after `--` go to binary
#
# Requires the binary to exist at $BINARY (default: cmake-build-release/poker_ppo).
# Override with POKER_PPO_BINARY env var if needed.
#

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

BINARY="${POKER_PPO_BINARY:-./cmake-build-release/poker_ppo}"
RUNS_DIR="./runs"
TS_REGEX='^[0-9]{8}_[0-9]{6}$'

force_fresh=0
force_resume=""
extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-fresh)
            force_fresh=1
            shift
            ;;
        --force-resume)
            shift
            if [[ $# -eq 0 ]]; then
                echo "--force-resume requires a run directory" >&2
                exit 2
            fi
            force_resume="$1"
            shift
            ;;
        --)
            shift
            extra_args=("$@")
            break
            ;;
        -h|--help)
            sed -n '3,22p' "$0"
            exit 0
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

if [[ ! -x "$BINARY" ]]; then
    echo "error: binary not found or not executable: $BINARY" >&2
    echo "       build first: cmake --build cmake-build-release -j" >&2
    exit 1
fi

resume_with() {
    local dir="$1"
    echo "==> resuming run: $dir"
    exec "$BINARY" --resume "$dir" "${extra_args[@]}"
}

start_fresh() {
    local reason="$1"
    echo "==> starting fresh run ($reason)"
    exec "$BINARY" "${extra_args[@]}"
}

# Explicit resume overrides everything else.
if [[ -n "$force_resume" ]]; then
    if [[ ! -d "$force_resume" ]]; then
        echo "error: --force-resume target not a directory: $force_resume" >&2
        exit 1
    fi
    resume_with "$force_resume"
fi

if [[ "$force_fresh" -eq 1 ]]; then
    start_fresh "--force-fresh"
fi

# Auto-detect: most recent timestamped run dir.
if [[ ! -d "$RUNS_DIR" ]]; then
    start_fresh "no runs/ directory yet"
fi

latest_run=$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
             | grep -E "$TS_REGEX" \
             | sort \
             | tail -n 1 || true)

if [[ -z "$latest_run" ]]; then
    start_fresh "no prior runs found"
fi

latest_path="$RUNS_DIR/$latest_run"

if [[ -f "$latest_path/FINISHED" ]]; then
    echo "    last run $latest_run already completed ($(cat "$latest_path/FINISHED" | head -1))"
    start_fresh "previous run finished"
fi

# No FINISHED marker — either in-progress (shouldn't happen if start.sh is the
# only launch path) or interrupted. Require at least one checkpoint to resume.
ckpt_dir="$latest_path/ckpt"
if [[ ! -d "$ckpt_dir" ]] || ! compgen -G "$ckpt_dir/update_*.pt" > /dev/null; then
    echo "    $latest_run has no checkpoint to resume from"
    start_fresh "previous run died before first checkpoint"
fi

# Show what we're resuming.
latest_ckpt=$(ls -1 "$ckpt_dir"/update_*.pt 2>/dev/null \
              | sed -E 's/.*update_([0-9]+)\.pt/\1/' \
              | sort -n \
              | tail -n 1)
echo "    latest checkpoint: update_${latest_ckpt}.pt"

resume_with "$latest_path"
