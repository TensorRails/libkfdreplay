#!/usr/bin/env bash
# Build the HIP toy and run one cold cycle.
#   examples/hip_toy/run.sh [minN] [residue]     e.g. run.sh 16384 2  (wptr just past one ring lap)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
[ -x "$here/toy" ] || hipcc -O2 -o "$here/toy" "$here/toy.hip"
exec "$here/../../scripts/cold_cycle.sh" -t "${WD:-120}" -d "${DIR:-$PWD/kfdr-toy}" -- "$here/toy" cold "${1:-0}" "${2:-1}"
