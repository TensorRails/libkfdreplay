#!/usr/bin/env bash
# shellcheck disable=SC2046  # $(images): checkpoint file names never contain spaces
# One cold checkpoint/restore cycle of a program under DMTCP + libkfdreplay.
#
#   scripts/cold_cycle.sh [options] -- <command> [args...]
#
# 1. launch:  LD_PRELOAD=libkfdreplay.so dmtcp_launch <command>. The program
#             warms up, calls kfdr_checkpoint(1) and dmtcp_checkpoint(), and the
#             source process exits after the image is written (see examples/).
# 2. check:   images exist, no process of the source tree is left alive.
# 3. restart: dmtcp_restart the images; the program calls kfdr_restore() and
#             runs its checks, printing "RESULT PASS" or "RESULT FAIL".
#
# Every phase runs under a watchdog: on timeout the whole tree (launcher,
# restored processes, coordinator) is killed, optionally after dumping gdb
# stacks. A plain `timeout dmtcp_launch` is not enough: it kills only the
# launcher, and the application and its children keep running.
#
# Options:
#   -d DIR      work/checkpoint directory (default: ./kfdr-cycle-<time>)
#   -t SEC      watchdog timeout per phase (default: 600)
#   -k PATTERN  extra process pattern (pkill -f) to kill on timeout/cleanup,
#               e.g. 'VLLM::EngineCor'; may be repeated
#   -g          on timeout, dump stacks of matching processes with gdb first
#   -e          evict the images from the page cache before restart (cold read)
#   -c N        chain: restart up to N+1 times, the program re-checkpoints itself
#               N times (writes N to DIR/chain_left; see examples/pytorch)
#   -K          keep the images after a PASS (default: delete them)
#   -L          launch only (stop after the checkpoint is written)
#   -R          restart only (restart existing images in DIR)
#   -E VAR=VAL  extra environment for the launched program, e.g. -E KFDR_NOCOPY=1
#               (DMTCP keeps the launch environment across restart)
#
# Environment: see scripts/env.sh (KFDR_LIB, DMTCP_ROOT).
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/env.sh
. "$here/env.sh"

DIR="" TMO=600 GDB=0 EVICT=0 CHAIN=0 KEEP=0 MODE=both
PATTERNS=() EXTRA_ENV=()
while getopts "d:t:k:gec:KLRE:h" o; do
    case $o in
    d) DIR=$OPTARG ;;
    t) TMO=$OPTARG ;;
    k) PATTERNS+=("$OPTARG") ;;
    g) GDB=1 ;;
    e) EVICT=1 ;;
    c) CHAIN=$OPTARG ;;
    K) KEEP=1 ;;
    L) MODE=launch ;;
    R) MODE=restart ;;
    E) EXTRA_ENV+=("$OPTARG") ;;
    *) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
if [ "$MODE" != restart ] && [ $# -eq 0 ]; then
    echo "usage: $0 [options] -- <command> [args...]   (-h for help)" >&2
    exit 2
fi
kfdr_check_env || exit 2

DIR=${DIR:-$PWD/kfdr-cycle-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
# Default kill pattern: the script/binary the command runs (e.g. torch_graph.py).
for a in "$@"; do
    case $a in *.py | */*) PATTERNS+=("$(basename "$a")") ;; esac
done
[ ${#PATTERNS[@]} -eq 0 ] && [ $# -gt 0 ] && PATTERNS+=("$(basename "$1")")

PORT=$((7000 + RANDOM % 2000))
launch_args=(--no-gzip --new-coordinator -p "$PORT" --ckptdir "$DIR")
[ -n "${DMTCP_FDKEEP:-}" ] && launch_args+=(--with-plugin "$DMTCP_FDKEEP")

now() { date +%s.%N; }

kill_tree() {
    for p in "${PATTERNS[@]}"; do pkill -9 -f -- "$p" 2>/dev/null; done
    pkill -9 -f -- "dmtcp_(launch|restart).*$DIR" 2>/dev/null
    pkill -9 -f -- "dmtcp_coordinator.*($PORT|$((PORT + 1)))" 2>/dev/null
    return 0
}

# watchdog <phase>: kill everything after $TMO s unless cancelled (kill $WDP).
watchdog() {
    (
        sleep "$TMO"
        echo "WATCHDOG $1 fired after ${TMO}s" | tee -a "$DIR/watchdog.txt" >&2
        if [ "$GDB" = 1 ] && command -v gdb >/dev/null; then
            for p in "${PATTERNS[@]}"; do
                for pid in $(pgrep -f -- "$p"); do
                    timeout 60 gdb -p "$pid" -batch -ex "thread apply all bt 16" >"$DIR/gdb_$1_$pid.txt" 2>&1
                done
            done
        fi
        kill_tree
    ) &
    WDP=$!
}
unwatch() { kill "$WDP" 2>/dev/null; wait "$WDP" 2>/dev/null; }

result_of() { grep -ho "RESULT [A-Z]*" "$@" 2>/dev/null | tail -1; }

images() { ls "$DIR"/ckpt_*.dmtcp 2>/dev/null; }

src_rc=- t_launch=0
if [ "$MODE" != restart ]; then
    rm -f "$DIR"/ckpt_*.dmtcp
    [ "$CHAIN" -gt 0 ] && echo "$CHAIN" >"$DIR/chain_left"
    t0=$(now)
    watchdog launch
    env "${EXTRA_ENV[@]}" LD_PRELOAD="$KFDR_LIB" "$DMTCP_BIN/dmtcp_launch" "${launch_args[@]}" "$@" >"$DIR/src.log" 2>&1
    src_rc=$?
    unwatch
    t_launch=$(echo "$(now) - $t0" | bc)
    if [ -z "$(images)" ]; then
        echo "cycle $DIR: NO CHECKPOINT (launch rc=$src_rc)"
        grep -E "\[kfdr\]|RESULT|Error|error" "$DIR/src.log" | tail -15
        kill_tree
        exit 3
    fi
    alive=0
    for p in "${PATTERNS[@]}"; do
        alive=$((alive + $(pgrep -f -- "$p" | grep -cv "^$$\$")))
    done
    if [ "$alive" -gt 0 ]; then
        echo "warning: $alive process(es) matching ${PATTERNS[*]} still alive after the source exited" >&2
    fi
    [ "$MODE" = launch ] && {
        echo "cycle $DIR: checkpoint written (launch rc=$src_rc, ${t_launch}s, $(du -cm --apparent-size $(images) | tail -1 | cut -f1) MiB)"
        kill_tree
        exit 0
    }
fi

[ -n "$(images)" ] || { echo "no images in $DIR" >&2; exit 3; }
if [ "$EVICT" = 1 ]; then
    sync
    python3 "$here/evict_page_cache.py" $(images)
fi
size=$(du -cm --apparent-size $(images) | tail -1 | cut -f1)

dst_rc=- res="" t_restart=0
for i in $(seq 0 "$CHAIN"); do
    PORT2=$((PORT + 1 + 2 * i))
    log="$DIR/dst.log"
    [ "$CHAIN" -gt 0 ] && log="$DIR/dst_$i.log"
    t2=$(now)
    watchdog "restart$i"
    # shellcheck disable=SC2086
    env $KFDR_RESTART_ENV "$DMTCP_BIN/dmtcp_restart" --no-strict-checking --new-coordinator -p "$PORT2" \
        --ckptdir "$DIR" $(images) >"$log" 2>&1
    dst_rc=$?
    unwatch
    t_restart=$(echo "$(now) - $t2" | bc)
    res=$(result_of "$log")
    [ -n "$res" ] && break
done
kill_tree

printf "cycle %s launch_rc=%s restart_rc=%s image=%sMiB launch+checkpoint=%.1fs restart+run=%.1fs %s\n" \
    "$(basename "$DIR")" "$src_rc" "$dst_rc" "$size" "$t_launch" "$t_restart" "${res:-NO RESULT}"
grep -hE "\[kfdr\] (REFUSE|restore|checkpoint done)|CRASH|different GPU" "$DIR"/src.log "$DIR"/dst*.log 2>/dev/null | tail -6
if [ "$res" = "RESULT PASS" ]; then
    [ "$KEEP" = 1 ] || rm -f "$DIR"/ckpt_*.dmtcp
    exit 0
fi
exit 1
