#!/usr/bin/env bash
# Gen-3 campaign, end to end, one command:
#
#   cd /poker_ppo/cmake-build-release && bash ../tools/vm_gen3.sh
#
# Shaped for 4x5090 / ~96 effective cores / 256GB RAM / 1TB disk. Phases, each
# gated on verifiable on-disk state (rerun after any death — finished
# phases are detected and skipped):
#   0  sanity ladder (kernels, early all-ins, river GPU, feature context,
#      fused turn GPU, safe gadget)
#   1  generation:
#        river + flop auxiliary banks start first; turn starts once the
#        short river bank finishes and overlaps the longer flop CPU lane.
#        GPUs 0..N-2 run pure fused turn solving; the last GPU hosts the
#        CPU turn lane's small oracle forwards. Mixing both lanes on every
#        GPU measured ~2x slower because persistent fused windows block the
#        CPU lane's latency-sensitive forwards.
#   2  transactional mix build (verified temp + atomic rename; sources kept)
#   3  training, concurrent mmap readers (shared Linux page cache):
#        g3_s1   — continued from the champion (1024x2)
#        g3_spec — fresh ReBeL-style 1536x6 GeLU+LayerNorm; the local
#                  architecture screen's best heldout arm
#   4  gates, sequential, paired seed 1234: a freshly measured champion
#      baseline plus CFR+/PCFR+ arms for both trained models. Fused B=1 play
#      remains off until the rental probe proves a latency win.
set -uo pipefail

effective_cores() {
    local online quota period limited
    online=$(nproc)
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max
        if [ "$quota" != max ] && [ "$period" -gt 0 ] 2>/dev/null; then
            limited=$(( quota / period ))
            [ "$limited" -ge 1 ] || limited=1
            [ "$limited" -lt "$online" ] && online=$limited
        fi
    fi
    echo "$online"
}

CORES=${CORES:-$(effective_cores)}
BIN=${BIN:-./rebel_hunl}
if [ -z "${CHAMPION:-}" ]; then
    if [ -s cand_f1.pt ]; then CHAMPION=cand_f1.pt
    elif [ -s ../cand_f1.pt ]; then CHAMPION=../cand_f1.pt
    else CHAMPION=cand_f1.pt
    fi
fi
CHAMPION_SIZE=42470807
TARGET_ROWS=${TARGET_ROWS:-5000000}
RIVER_TARGET=${RIVER_TARGET:-500000}
RIVER_EPOCHS=${RIVER_EPOCHS:-20}
RIVER_EPS=${RIVER_EPS:-25000}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-40}
TRAIN_SGD=${TRAIN_SGD:-4000}
GPU_CACHE=${GPU_CACHE:-500000}          # ~20GB on device; 5090 has 32GB
DISK_HEADROOM_GB=${DISK_HEADROOM_GB:-180}
FLOP_EPOCHS=${FLOP_EPOCHS:-25}
FLOP_EPS=${FLOP_EPS:-500}
FLOP_THREADS=${FLOP_THREADS:-40}
CHUNK_EPISODES=${CHUNK_EPISODES:-20000}
EPOCHS_PER_RUN=${EPOCHS_PER_RUN:-25}
GPU_FRAC=${GPU_FRAC:-0.3}
GPU_BATCH=${GPU_BATCH:-128}
# From rental_probe's pure-lane rates C (all CPU workers) and G (one fused
# GPU), balance completion with CPUSHARE=C/(C+(NGPU-1)*G).
CPUSHARE=${CPUSHARE:-0.15}
GPU_SUPPORT_THREADS=${GPU_SUPPORT_THREADS:-12}
T_TURN=${T_TURN:-120}
REPLAY_CAP=${REPLAY_CAP:-5600000}      # keeps the full ~5.54M mixed bank
GATE_HANDS=${GATE_HANDS:-20000}
GATE_THREADS=${GATE_THREADS:-$CORES}
GATE_MAX_REGRESSION=${GATE_MAX_REGRESSION:-0.35}
PRESSURE_HANDS=${PRESSURE_HANDS:-200}
PRESSURE_THREADS=${PRESSURE_THREADS:-16}
LATENCY_HANDS=${LATENCY_HANDS:-20}
ROOT_EPOCHS=${ROOT_EPOCHS:-5}
ROOT_EPISODES=${ROOT_EPISODES:-40}
ROOT_SGD=${ROOT_SGD:-1000}
ROOT_THREADS=${ROOT_THREADS:-$CORES}
SEED_BASE=${SEED_BASE:-0}
# Each shard owns a disjoint seed interval.  The trainer currently reduces
# several RNG seeds to uint32, so keep the real campaign below this stride
# and below 2^32 rather than relying on a hash collision not happening.
SEED_STRIDE=${SEED_STRIDE:-100000000}
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

# Pick an exact (epochs x episodes) chunk whenever possible without making
# an epoch larger than CHUNK_EPISODES.  If the remaining count has no useful
# factor, emit full-size epochs now and a small exact tail on the next pass.
# Sets the caller's `epochs` and `eps` variables.
plan_chunk() {  # <remaining>
    local remaining=$1 e max_e
    eps=$CHUNK_EPISODES
    epochs=$EPOCHS_PER_RUN
    [ $(( eps * epochs )) -le "$remaining" ] && return

    max_e=$EPOCHS_PER_RUN
    [ "$max_e" -gt "$remaining" ] && max_e=$remaining
    for ((e=max_e; e>=1; --e)); do
        if [ $(( remaining % e )) -eq 0 ] && \
           [ $(( remaining / e )) -le "$CHUNK_EPISODES" ]; then
            epochs=$e
            eps=$(( remaining / e ))
            return
        fi
    done

    epochs=$(( remaining / CHUNK_EPISODES ))
    [ "$epochs" -ge 1 ] || epochs=1
    eps=$CHUNK_EPISODES
}

check_chunk_plan() {  # <target> <seed-base>
    local target=$1 base=$2 total=0 last=-1 remaining made seed steps=0
    [ "$target" -gt 0 ] || die "plan-check target must be positive"
    while [ "$total" -lt "$target" ]; do
        remaining=$(( target - total ))
        plan_chunk "$remaining"
        made=$(( epochs * eps ))
        [ "$made" -gt 0 ] && [ "$made" -le "$remaining" ] \
            || die "invalid chunk ${epochs}x${eps} for remaining=$remaining"
        seed=$(( 7000 + base + total ))
        [ "$seed" -gt "$last" ] && [ "$seed" -le 4294967295 ] \
            || die "non-monotonic/out-of-range seed $seed"
        printf 'plan chunk=%d rows=%d seed=%d\n' "$steps" "$made" "$seed"
        total=$(( total + made ))
        last=$seed
        steps=$(( steps + 1 ))
        [ "$steps" -lt 10000 ] || die "chunk planner did not converge"
    done
    [ "$total" -eq "$target" ] || die "chunk plan ended at $total, want $target"
    echo "plan PASS: $steps chunks, $total rows"
}

if [ "${REBEL_CHUNK_PLAN_CHECK:-0}" = 1 ]; then
    check_chunk_plan "$TARGET_ROWS" "$SEED_BASE"
    exit 0
fi

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
    [ "$TARGET_ROWS" -lt "$SEED_STRIDE" ] \
        || die "TARGET_ROWS must be < SEED_STRIDE ($SEED_STRIDE)"
    while true; do
        have=$(rows "$OUT")
        [ "$have" -ge "$TARGET_ROWS" ] && break
        remaining=$(( TARGET_ROWS - have ))
        plan_chunk "$remaining"
        seed=$(( 7000 + SEED_BASE + have ))
        [ "$seed" -le 4294967295 ] \
            || die "derived seed $seed exceeds uint32; lower SEED_BASE/NGPU"
        planned=$(( epochs * eps ))
        echo "== chunk: $epochs x $eps episodes ($have/$TARGET_ROWS rows, seed=$seed)"
        REBEL_TF32=1 REBEL_PCFR=1 REBEL_ZERO_SUM=1 \
        REBEL_HIDDEN=1024 REBEL_LAYERS=2 REBEL_GELU_LN=0 \
        REBEL_RIVER_ROWS=0 REBEL_HARVEST=0 \
        REBEL_SGD_STEPS=0 REBEL_PROBE_K=0 \
        REBEL_SEED=$seed \
        REBEL_CKPT=gen_oracle.pt REBEL_DATA_OUT="$OUT" \
        REBEL_T_TURN_TRAIN=$T_TURN REBEL_THREADS=$TURN_THREADS \
        REBEL_GPU_TURN_BATCH=$GPU_BATCH REBEL_GPU_FRAC=$GPU_FRAC \
        $BIN train_turn "$epochs" "$eps" > "${OUT%.bin}_chunk$have.log" 2>&1 &
        finish "$!" "${OUT%.bin}_chunk$have.log" "$epochs" "turn ${OUT}" \
            || exit 1
        aligned "$OUT" || die "$OUT misaligned after chunk at row $have"
        after=$(rows "$OUT")
        [ "$after" -gt "$have" ] \
            || die "$OUT chunk at row $have produced no rows"
        [ $(( after - have )) -le "$planned" ] \
            || die "$OUT grew by more than its $planned-row chunk (second writer?)"
    done
    echo "== turn shard done: $OUT $(rows "$OUT") rows"
    exit 0
fi

# ── gates ─────────────────────────────────────────────────────────────
[ -x "$BIN" ] || die "$BIN not found — run from cmake-build-release"
command -v flock >/dev/null 2>&1 || die "flock is required for campaign locking"
exec 9>.vm_gen3.lock
flock -n 9 || die "another vm_gen3 campaign owns .vm_gen3.lock"
[ "$TARGET_ROWS" -gt 0 ] || die "TARGET_ROWS must be positive"
[ "$CORES" -ge 1 ] || die "CORES must be positive"
[ "$DISK_HEADROOM_GB" -ge 0 ] \
    || die "DISK_HEADROOM_GB must be non-negative"
[ $(( RIVER_EPOCHS * RIVER_EPS )) -eq "$RIVER_TARGET" ] \
    || die "RIVER_EPOCHS*RIVER_EPS must equal RIVER_TARGET exactly"
[ "$CHUNK_EPISODES" -gt 0 ] || die "CHUNK_EPISODES must be positive"
[ "$EPOCHS_PER_RUN" -gt 0 ] || die "EPOCHS_PER_RUN must be positive"
[ "$TARGET_ROWS" -lt "$SEED_STRIDE" ] \
    || die "TARGET_ROWS must be < SEED_STRIDE ($SEED_STRIDE)"
[ "$(stat -c%s "$CHAMPION" 2>/dev/null || echo 0)" -eq "$CHAMPION_SIZE" ] \
    || die "$CHAMPION missing or wrong size (want $CHAMPION_SIZE)"
[ -f ../HandRanks.dat ] || [ -f HandRanks.dat ] \
    || [ -f ../Game/Utility/HandRanks.dat ] \
    || die "HandRanks.dat not found — scp it (untracked in git)"
pgrep -f 'rebel_hun[l]' >/dev/null \
    && die "rebel_hunl already running — kill strays first"
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NGPU" -ge 1 ] || NGPU=1

# Refuse to resume shards made by older launchers/target semantics: v2 fixed
# modulo seed reuse; v3 also aligns ReBeL iteration weighting.
# repeated every default 500k chunk, so those rows are not salvageable by
# merely continuing with the corrected schedule.
SEED_SCHEME_FILE=.vm_gen3_seed_scheme
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required for provenance"
champion_sha=$(sha256sum "$CHAMPION" | awk '{print $1}')
binary_sha=$(sha256sum "$BIN" | awk '{print $1}')
SEED_SCHEME="absolute-row-v3 champion=$champion_sha binary=$binary_sha t_turn=$T_TURN pcfr=1 quad=1 root_weight=matched refresh=5 data_v=3 row=$ROW ngpu=$NGPU cpushare=$CPUSHARE gpu_batch=$GPU_BATCH seed_base=$SEED_BASE seed_stride=$SEED_STRIDE"
has_data=0
compgen -G 'turn_g*.bin' >/dev/null && has_data=1
[ -e mix.bin ] || [ -e river_g3.bin ] || [ -e flop_g3.bin ] && has_data=1
if [ "$has_data" -eq 1 ]; then
    [ -f "$SEED_SCHEME_FILE" ] && \
    [ "$(cat "$SEED_SCHEME_FILE")" = "$SEED_SCHEME" ] \
        || die "existing campaign provenance differs; move it aside and regenerate"
else
    printf '%s\n' "$SEED_SCHEME" > "$SEED_SCHEME_FILE" \
        || die "cannot write $SEED_SCHEME_FILE"
fi
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
# Transactional mixing retains every source while writing a complete temp
# copy.  At 5M turn rows the default peak is ~444GB; require extra room for
# logs, checkpoints, allocator spill, and filesystem accounting.
need_gb=$(( 2 * TARGET_ROWS * ROW / 1000000000 + DISK_HEADROOM_GB ))
[ "$(rows mix.bin)" -gt 0 ] || [ "$free_gb" -ge "$need_gb" ] \
    || die "need ~${need_gb}GB free, have ${free_gb}GB"

# ── phase 0: sanity ──────────────────────────────────────────────────
echo "== phase 0: sanity"
kernel_line=$($BIN kernels 2>&1 | tail -1) || die "kernels FAIL"
echo "$kernel_line"
echo "$kernel_line" | grep -Eq \
    'root-weight err .*topology-key=exact +PASS$' \
    || die "kernels FAIL or stale rebel_hunl binary"
$BIN allin_response_check 2>&1 | tail -1 | grep -q PASS \
    || die "allin_response_check FAIL"
$BIN gpu_check 4 60 2>&1 | tail -1 | grep -q PASS || die "gpu_check FAIL"
flop_line=$($BIN flopctx_check 10 2>&1 | tail -1) || die "flopctx FAIL"
echo "$flop_line"
echo "$flop_line" | grep -Eq 'range-scale-invariance .* PASS$' \
    || die "flopctx FAIL or stale rebel_hunl binary"
turn_line=$(REBEL_CKPT="$CHAMPION" REBEL_PCFR=1 REBEL_TF32=1 \
    $BIN turn_gpu_check 4 40 2>&1 | tail -1) \
    || die "fused PCFR+ turn_gpu_check FAIL"
echo "$turn_line"
[ "$turn_line" = "turn_gpu_check: PASS" ] \
    || die "fused PCFR+ turn_gpu_check FAIL or stale rebel_hunl binary"
$BIN gadget_check 400 2>&1 | tail -1 | grep -q PASS \
    || die "gadget_check FAIL"
echo "   all PASS"

# ── phase 1a: auxiliary river/flop banks ─────────────────────────────
# sgd_steps=0 is a trainer contract: these copies are loaded as immutable
# generation oracles and are not checkpointed at epoch end.  Refresh them on
# every launcher start so a stale/corrupt copy cannot survive a prior crash.
cp "$CHAMPION" gen_oracle.pt || die "cannot stage gen_oracle.pt"
cp "$CHAMPION" flop_oracle.pt || die "cannot stage flop_oracle.pt"
cores=$CORES

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
    echo "== flop lane: ~$FLOP_TARGET rows on CPUs"
    # The last GPU is reserved for latency-sensitive CPU-lane forwards; it
    # may share those small forwards with the CPU turn lane, but never runs
    # the persistent fused turn kernel.
    CUDA_VISIBLE_DEVICES=$(( NGPU - 1 )) \
    REBEL_TF32=1 REBEL_PCFR=1 REBEL_ZERO_SUM=1 \
    REBEL_HIDDEN=1024 REBEL_LAYERS=2 REBEL_GELU_LN=0 \
    REBEL_SEED=101 REBEL_CKPT=flop_oracle.pt \
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

[ -n "$RIVER_PID" ] && { finish "$RIVER_PID" river_g3.log "$RIVER_EPOCHS" \
    "river lane" || die "river lane failed"; }
aligned river_g3.bin || die "river_g3.bin misaligned"
[ "$(rows river_g3.bin)" -eq "$RIVER_TARGET" ] \
    || die "river_g3.bin has $(rows river_g3.bin) rows; want $RIVER_TARGET"

# ── phase 1b: turn bank, dedicated resource lanes ────────────────────
if [ "$(rows mix.bin)" -eq 0 ]; then
    turn_targets=(); turn_threads=(); turn_fracs=(); turn_batches=()
    if [ "$NGPU" -gt 1 ]; then
        cpu_rows=$(awk -v t="$TARGET_ROWS" -v s="$CPUSHARE" \
                       'BEGIN{printf "%d", t*s}')
        [ "$cpu_rows" -ge 0 ] && [ "$cpu_rows" -le "$TARGET_ROWS" ] \
            || die "CPUSHARE=$CPUSHARE produced invalid CPU target $cpu_rows"
        gpu_total=$(( TARGET_ROWS - cpu_rows ))
        gpu_lanes=$(( NGPU - 1 ))
        gpu_base=$(( gpu_total / gpu_lanes ))
        gpu_extra=$(( gpu_total % gpu_lanes ))
        for g in $(seq 0 $(( gpu_lanes - 1 ))); do
            lane_rows=$gpu_base
            [ "$g" -lt "$gpu_extra" ] && lane_rows=$(( lane_rows + 1 ))
            turn_targets[$g]=$lane_rows
            turn_threads[$g]=$GPU_SUPPORT_THREADS
            turn_fracs[$g]=1.0
            turn_batches[$g]=$GPU_BATCH
        done
        g=$(( NGPU - 1 ))
        flop_reserve=0
        [ -n "$FLOP_PID" ] && flop_reserve=$FLOP_THREADS
        cpu_threads=$(( cores - GPU_SUPPORT_THREADS * gpu_lanes - flop_reserve - 4 ))
        [ "$cpu_threads" -ge 8 ] || cpu_threads=8
        turn_targets[$g]=$cpu_rows
        turn_threads[$g]=$cpu_threads
        turn_fracs[$g]=0.0
        turn_batches[$g]=0
    else
        turn_targets[0]=$TARGET_ROWS
        turn_threads[0]=$(( cores - 8 ))
        [ "${turn_threads[0]}" -ge 4 ] || turn_threads[0]=4
        turn_fracs[0]=$GPU_FRAC
        turn_batches[0]=$GPU_BATCH
    fi

    turn_done=1
    for g in $(seq 0 $(( NGPU - 1 ))); do
        [ "$(rows "turn_g$g.bin")" -lt "${turn_targets[$g]}" ] \
            && turn_done=0
    done
    if [ "$turn_done" -eq 0 ]; then
        echo "== turn lane: dedicated layout"
        for g in $(seq 0 $(( NGPU - 1 ))); do
            echo "   gpu $g: ${turn_targets[$g]} rows, ${turn_threads[$g]} threads," \
                 "gpu_frac=${turn_fracs[$g]} batch=${turn_batches[$g]}"
        done
        tpids=()
        for g in $(seq 0 $(( NGPU - 1 ))); do
            [ "$(rows "turn_g$g.bin")" -ge "${turn_targets[$g]}" ] && continue
            CUDA_VISIBLE_DEVICES=$g GEN3_TURN_CHILD=1 \
            TARGET_ROWS=${turn_targets[$g]} TURN_THREADS=${turn_threads[$g]} \
            GPU_FRAC=${turn_fracs[$g]} GPU_BATCH=${turn_batches[$g]} \
            SEED_BASE=$(( SEED_BASE + g * SEED_STRIDE )) \
            SEED_STRIDE=$SEED_STRIDE BIN="$BIN" CHAMPION="$CHAMPION" \
            bash "$0" "turn_g$g.bin" > "turn_launcher_g$g.log" 2>&1 &
            tpids+=("$!")
        done
        tfail=0
        for p in "${tpids[@]}"; do wait "$p" || tfail=1; done
        [ "$tfail" -eq 0 ] || die "a turn shard failed (turn_launcher_g*.log)"
    fi
    echo "== turn lane: complete"
    turn_sum=0
    for g in $(seq 0 $(( NGPU - 1 ))); do
        got=$(rows "turn_g$g.bin")
        [ "$got" -eq "${turn_targets[$g]}" ] \
            || die "turn_g$g.bin has $got rows; want exactly ${turn_targets[$g]}"
        turn_sum=$(( turn_sum + got ))
    done
    [ "$turn_sum" -eq "$TARGET_ROWS" ] \
        || die "turn shards total $turn_sum rows; want $TARGET_ROWS"
fi

[ -n "$FLOP_PID" ] && { finish "$FLOP_PID" flop_g3.log "$FLOP_EPOCHS" \
    "flop lane" || die "flop lane failed"; }
aligned flop_g3.bin || die "flop_g3.bin misaligned"

# Runtime guard for the sgd_steps=0 immutable-oracle contract. An older
# binary would rewrite these paths after every epoch (and all turn children
# would race on gen_oracle.pt); refuse to build a mix if either byte changed.
cmp -s "$CHAMPION" gen_oracle.pt \
    || die "gen_oracle.pt was rewritten — rebuild rebel_hunl with zero-SGD save suppression"
cmp -s "$CHAMPION" flop_oracle.pt \
    || die "flop_oracle.pt was rewritten — rebuild rebel_hunl with zero-SGD save suppression"

# ── phase 2: transactional mix ───────────────────────────────────────
river_rows=$(rows river_g3.bin)
flop_rows=$(rows flop_g3.bin)
expected_mix_rows=$(( TARGET_ROWS + river_rows + flop_rows ))

# Header equality catches a same-row-size dataset from a foreign
# featurizer/version before concatenation.  All sources remain intact until
# a fully verified temp file is atomically renamed into place.
header_ref=turn_g0.bin
for g in $(seq 0 $(( NGPU - 1 ))); do
    f="turn_g$g.bin"
    [ -f "$f" ] || die "$f missing before mix"
    aligned "$f" || die "$f misaligned"
    cmp -n 16 "$header_ref" "$f" >/dev/null \
        || die "$f header differs from $header_ref"
done
for f in river_g3.bin flop_g3.bin; do
    [ -f "$f" ] || die "$f missing before mix"
    aligned "$f" || die "$f misaligned"
    cmp -n 16 "$header_ref" "$f" >/dev/null \
        || die "$f header differs from $header_ref"
done

if [ "$(rows mix.bin)" -ne "$expected_mix_rows" ]; then
    [ ! -e mix.bin ] || die "mix.bin has $(rows mix.bin) rows; expected $expected_mix_rows — move it aside"
    MIX_TMP=mix.bin.tmp
    rm -f "$MIX_TMP"
    echo "== building transactional mix.bin ($expected_mix_rows rows)"
    cp "$header_ref" "$MIX_TMP" || die "cannot create $MIX_TMP"
    for g in $(seq 1 $(( NGPU - 1 ))); do
        tail -c +17 "turn_g$g.bin" >> "$MIX_TMP" \
            || die "failed appending turn_g$g.bin"
    done
    tail -c +17 river_g3.bin >> "$MIX_TMP" \
        || die "failed appending river_g3.bin"
    tail -c +17 flop_g3.bin >> "$MIX_TMP" \
        || die "failed appending flop_g3.bin"
    sync "$MIX_TMP" || die "cannot sync $MIX_TMP"
    aligned "$MIX_TMP" || die "$MIX_TMP misaligned"
    cmp -n 16 "$header_ref" "$MIX_TMP" >/dev/null \
        || die "$MIX_TMP header changed during build"
    [ "$(rows "$MIX_TMP")" -eq "$expected_mix_rows" ] \
        || die "$MIX_TMP row count $(rows "$MIX_TMP") != $expected_mix_rows"
    mv "$MIX_TMP" mix.bin || die "cannot atomically publish mix.bin"
fi
aligned mix.bin || die "mix.bin misaligned"
cmp -n 16 "$header_ref" mix.bin >/dev/null || die "mix.bin header mismatch"
[ "$(rows mix.bin)" -eq "$expected_mix_rows" ] \
    || die "mix.bin row count $(rows mix.bin) != $expected_mix_rows"
echo "== mix: $(rows mix.bin) rows (turn=$TARGET_ROWS river=$river_rows flop=$flop_rows)"

# ── phase 3: mmap training arms (concurrent when >=2 GPUs) ─────────
train_one() {  # <ckpt> <seed> <gpu> <extra-env...>
    local ck=$1 sd=$2 gpu=$3
    shift 3
    grep -Eq "epoch +${TRAIN_EPOCHS} +replay=" "train_${ck%.pt}.log" \
        2>/dev/null && {
        [ -s "$ck" ] || die "$ck log is complete but checkpoint is missing"
        echo "== $ck: already trained"
        return 0
    }
    echo "== training $ck"
    # Offline data is a fixed corpus, not an evolving self-play stream. The
    # device cache walks a shuffled permutation without replacement: 500k
    # exposes the complete ~5.54M mix every ~11 epochs (the old 60k random
    # redraws reached only about 38% of 5M rows over 40 epochs).
    env CUDA_VISIBLE_DEVICES="$gpu" REBEL_TF32=1 \
        REBEL_MMAP_DATA=1 REBEL_GPU_CACHE=$GPU_CACHE \
        REBEL_GPU_CACHE_VERIFY=1 REBEL_CIRCULAR=0 \
        REBEL_ZERO_SUM=1 REBEL_HUBER_DELTA=1 \
        REBEL_HIDDEN=1024 REBEL_LAYERS=2 REBEL_GELU_LN=0 \
        REBEL_SEED="$sd" \
        REBEL_CKPT="$ck" REBEL_DATA_IN=mix.bin \
        REBEL_REPLAY_CAP=$REPLAY_CAP REBEL_PROBE_K=0 \
        REBEL_SGD_STEPS=$TRAIN_SGD REBEL_BATCH=2048 "$@" \
        $BIN train_turn "$TRAIN_EPOCHS" 0 > "train_${ck%.pt}.log" 2>&1 || {
            tail -20 "train_${ck%.pt}.log" >&2
            die "$ck training process failed"
        }
    grep -Eq "epoch +${TRAIN_EPOCHS} +replay=" "train_${ck%.pt}.log" \
        || die "$ck training did not finish (train_${ck%.pt}.log)"
    grep -m1 'replay=' "train_${ck%.pt}.log" | grep -q 'replay= *0 ' \
        && die "$ck trained on EMPTY replay"
    [ -s "$ck" ] \
        || die "$ck never beat its baseline; no checkpoint was published"
    return 0
}
[ -f g3_s1.pt ] || cp "$CHAMPION" g3_s1.pt
if [ "$NGPU" -ge 2 ]; then
    train_one g3_s1.pt 301 0 REBEL_LR=2e-4 REBEL_LR_FINAL=5e-5 &
    train_pids=("$!")
    train_one g3_spec.pt 302 1 REBEL_LR=3e-4 REBEL_LR_FINAL=3e-5 \
        REBEL_BATCH=1024 REBEL_HIDDEN=1536 REBEL_LAYERS=6 REBEL_GELU_LN=1 &
    train_pids+=("$!")
    train_fail=0
    for p in "${train_pids[@]}"; do wait "$p" || train_fail=1; done
    [ "$train_fail" -eq 0 ] || die "a concurrent mmap training arm failed"
else
    train_one g3_s1.pt 301 0 REBEL_LR=2e-4 REBEL_LR_FINAL=5e-5
    train_one g3_spec.pt 302 0 REBEL_LR=3e-4 REBEL_LR_FINAL=3e-5 \
        REBEL_BATCH=1024 REBEL_HIDDEN=1536 REBEL_LAYERS=6 REBEL_GELU_LN=1
fi

# ── phase 4: gates, sequential, paired seed 1234 ────────────────────
gate_one() {  # <name> <ckpt> <extra-env...>
    local name=$1 ck=$2
    local gate_hands=${GATE_THIS_HANDS:-$GATE_HANDS}
    local gate_threads=${GATE_THIS_THREADS:-$GATE_THREADS}
    shift 2
    grep -q 'LBR vs ReBeL agent:' "gate_$name.log" 2>/dev/null && {
        echo "== gate $name: done"
        return 0
    }
    rm -f "gate_$name.csv" "gate_$name.csv".*
    echo "== gate $name ($gate_hands hands @ $gate_threads thr)"
    env -u REBEL_BLUEPRINT -u REBEL_BLUEPRINT_ONLY \
        REBEL_TF32=1 REBEL_PCFR=0 REBEL_GADGET=0 REBEL_GPU_TURN=0 \
        REBEL_GADGET_ALT_EXACT=1 REBEL_GADGET_ALT_MODE=0 \
        REBEL_GADGET_DELTA=0 REBEL_T_ALT=200 \
        REBEL_ZERO_SUM=1 REBEL_HIDDEN=1024 REBEL_LAYERS=2 REBEL_GELU_LN=0 \
        REBEL_FLOP=1 REBEL_PREFLOP=1 REBEL_T_PREFLOP=40 REBEL_T_FLOP=60 \
        REBEL_T_TURN=240 REBEL_T_RIVER=800 REBEL_PF_SAMPLES=64 "$@" \
        REBEL_CKPT="$ck" REBEL_SEED=1234 REBEL_LBR_LOG="gate_$name.csv" \
        $BIN lbr "$gate_hands" "$gate_threads" > "gate_$name.log" 2>&1
    grep -q 'LBR vs ReBeL agent:' "gate_$name.log" \
        || die "gate $name did not finish"
}
gate_one g3champ_cfr     "$CHAMPION"
gate_one g3s1_cfr        g3_s1.pt
gate_one g3s1_pcfr       g3_s1.pt REBEL_PCFR=1
gate_one g3s1_cfr_gadget g3_s1.pt REBEL_GADGET=1
gate_one g3spec_cfr      g3_spec.pt REBEL_HIDDEN=1536 REBEL_LAYERS=6 \
    REBEL_GELU_LN=1
gate_one g3spec_pcfr     g3_spec.pt REBEL_PCFR=1 \
    REBEL_HIDDEN=1536 REBEL_LAYERS=6 \
    REBEL_GELU_LN=1

# Pick on the trustworthy analytic LBR bound. The tolerance is deliberately
# paired to the freshly remeasured champion and defaults to ~1.6 standard
# errors for 20k HUNL hands; a rental can tighten it by raising GATE_HANDS.
gate_score() {
    grep 'LBR vs ReBeL agent:' "gate_$1.log" | tail -1 | awk '{print $5}'
}
champ_score=$(gate_score g3champ_cfr)
best_name=g3s1_cfr
best_score=$(gate_score "$best_name")
for n in g3s1_pcfr g3spec_cfr g3spec_pcfr; do
    s=$(gate_score "$n")
    if awk -v a="$s" -v b="$best_score" 'BEGIN{exit !(a < b)}'; then
        best_name=$n
        best_score=$s
    fi
done
awk -v b="$best_score" -v c="$champ_score" -v m="$GATE_MAX_REGRESSION" \
    'BEGIN{exit !(b <= c + m)}' \
    || die "best candidate $best_name=$best_score regresses champion=$champ_score by more than $GATE_MAX_REGRESSION bb/hand"

case "$best_name" in
    g3s1_cfr)   selected_ckpt=g3_s1.pt; selected_pcfr=0; selected_arch=base ;;
    g3s1_pcfr)  selected_ckpt=g3_s1.pt; selected_pcfr=1; selected_arch=base ;;
    g3spec_cfr) selected_ckpt=g3_spec.pt; selected_pcfr=0; selected_arch=spec ;;
    g3spec_pcfr) selected_ckpt=g3_spec.pt; selected_pcfr=1; selected_arch=spec ;;
    *) die "unknown selected gate $best_name" ;;
esac
if [ "$selected_arch" = spec ]; then
    selected_env=(REBEL_HIDDEN=1536 REBEL_LAYERS=6 REBEL_GELU_LN=1)
else
    selected_env=(REBEL_HIDDEN=1024 REBEL_LAYERS=2 REBEL_GELU_LN=0)
fi
cp "$selected_ckpt" g3_selected.pt || die "cannot publish g3_selected.pt"
printf 'gate=%s analytic_lbr=%s champion_lbr=%s pcfr=%s arch=%s actions=0,1,7,13\n' \
    "$best_name" "$best_score" "$champ_score" "$selected_pcfr" \
    "$selected_arch" > g3_selected.meta

# Phase 5: Algorithm-1 closure from the real game root. This loads the full
# mixed corpus, then lets the selected model generate its own preflop->flop
# query distribution while it fine-tunes. It is intentionally after model
# selection: using the old champion here would leave the deployed winner on
# another model's leaf distribution.
if ! grep -Eq "epoch +${ROOT_EPOCHS} +replay=" root_refine.log 2>/dev/null; then
    [ ! -e root_refine.bin ] && [ ! -e g3_final.pt ] \
        || die "partial root refinement exists; move root_refine.bin, root_refine.log and g3_final.pt aside before retry"
    cp g3_selected.pt g3_final.pt || die "cannot stage g3_final.pt"
    echo "== phase 5: selected-model root refinement ($ROOT_EPOCHS x $ROOT_EPISODES)"
    env REBEL_TF32=1 REBEL_MMAP_DATA=1 REBEL_GPU_CACHE=$GPU_CACHE \
        REBEL_GPU_CACHE_VERIFY=1 \
        REBEL_CIRCULAR=0 \
        REBEL_ZERO_SUM=1 REBEL_HUBER_DELTA=1 REBEL_PCFR=1 \
        "${selected_env[@]}" REBEL_SEED=401 REBEL_CKPT=g3_final.pt \
        REBEL_DATA_IN=mix.bin REBEL_DATA_OUT=root_refine.bin \
        REBEL_REPLAY_CAP=$REPLAY_CAP REBEL_PROBE_K=0 \
        REBEL_THREADS=$ROOT_THREADS REBEL_T_PREFLOP=40 REBEL_PF_SAMPLES=64 \
        REBEL_T_FLOP=60 REBEL_T_TURN_TRAIN=$T_TURN REBEL_HARVEST=2 \
        REBEL_SGD_STEPS=$ROOT_SGD REBEL_BATCH=1024 REBEL_LR=5e-5 \
        REBEL_LR_FINAL=1e-5 \
        $BIN train_root "$ROOT_EPOCHS" "$ROOT_EPISODES" \
        > root_refine.log 2>&1 || {
            tail -20 root_refine.log >&2
            die "root refinement failed"
        }
fi
grep -Eq "epoch +${ROOT_EPOCHS} +replay=" root_refine.log \
    || die "root refinement did not finish"
[ -s g3_final.pt ] || die "root refinement produced no checkpoint"

GATE_THIS_HANDS=$GATE_HANDS GATE_THIS_THREADS=$GATE_THREADS \
    gate_one g3final_cfr g3_final.pt "${selected_env[@]}"
GATE_THIS_HANDS=$GATE_HANDS GATE_THIS_THREADS=$GATE_THREADS \
    gate_one g3final_pcfr g3_final.pt REBEL_PCFR=1 "${selected_env[@]}"
final_cfr=$(gate_score g3final_cfr)
final_pcfr=$(gate_score g3final_pcfr)
final_name=g3final_cfr
final_score=$final_cfr
if awk -v a="$final_pcfr" -v b="$final_cfr" 'BEGIN{exit !(a < b)}'; then
    final_name=g3final_pcfr
    final_score=$final_pcfr
fi
awk -v f="$final_score" -v c="$champ_score" -v m="$GATE_MAX_REGRESSION" \
    'BEGIN{exit !(f <= c + m)}' \
    || die "root-refined model=$final_score fails champion tolerance $champ_score+$GATE_MAX_REGRESSION"

# Small, expensive diagnostic: exercise target responses to both pot raises
# and all-ins on every street. Completion plus the allin_response_check above
# is the coverage gate; exploitability acceptance remains the paired analytic
# LBR threshold, not an arbitrary fold-frequency target.
GATE_THIS_HANDS=$PRESSURE_HANDS GATE_THIS_THREADS=$PRESSURE_THREADS \
    gate_one g3final_pressure g3_final.pt REBEL_FOLD_PROBE=1 \
        REBEL_SHOVE_PROBE=1 REBEL_ALLIN_MC_SAMPLES=1024 "${selected_env[@]}"
grep -q 'fold-rate to an all-in' gate_g3final_pressure.log \
    || die "pressure gate did not exercise shove responses"

final_pcfr=0
[ "$final_name" = g3final_pcfr ] && final_pcfr=1
printf 'gate=%s analytic_lbr=%s champion_lbr=%s pcfr=%s arch=%s actions=0,1,7,13 root_rows=%s\n' \
    "$final_name" "$final_score" "$champ_score" "$final_pcfr" \
    "$selected_arch" "$(rows root_refine.bin)" > g3_final.meta

printf 'bench %s\n' "$LATENCY_HANDS" | env REBEL_TF32=1 \
    REBEL_CKPT=g3_final.pt REBEL_PCFR=$final_pcfr REBEL_GADGET=0 \
    REBEL_GPU_TURN=0 REBEL_PREFLOP=1 REBEL_FLOP=1 \
    REBEL_T_PREFLOP=40 REBEL_T_FLOP=60 REBEL_T_TURN=240 \
    REBEL_T_RIVER=800 REBEL_PF_SAMPLES=64 "${selected_env[@]}" \
    $BIN play_server > g3_latency.jsonl 2> g3_latency.err \
    || die "play-server latency benchmark failed"
grep -q '"latency"' g3_latency.jsonl \
    || die "play-server latency benchmark produced no street metrics"

echo ""
echo "== GEN-3 RESULTS ========================================="
for n in g3champ_cfr g3s1_cfr g3s1_pcfr g3s1_cfr_gadget \
         g3spec_cfr g3spec_pcfr g3final_cfr g3final_pcfr; do
    echo "  $n  $(grep 'LBR vs ReBeL agent:' "gate_$n.log" 2>/dev/null || echo INCOMPLETE)"
done
echo "selected: $final_name ($final_score bb/hand), metadata g3_final.meta"
