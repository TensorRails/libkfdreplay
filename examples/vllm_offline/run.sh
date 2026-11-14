#!/usr/bin/env bash
# vLLM offline engine in sleep mode through a cold cycle; answers compared token by token
# with a reference run of the same configuration.
#   examples/vllm_offline/run.sh [inprocess|mp]
# Env: KFDR_EX_MODEL (default Qwen/Qwen3-0.6B), KFDR_EX_GPU_UTIL (default 0.05), EVICT=1 (cold image read).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:-inprocess}
script=$here/vllm_$mode.py
[ -f "$script" ] || { echo "mode must be inprocess or mp" >&2; exit 2; }
dir=${DIR:-$PWD/kfdr-vllm-$mode}
mkdir -p "$dir"
. "$here/../../scripts/env.sh"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
base=${BASELINE:-$dir/baseline.json}
if [ ! -f "$base" ]; then
    echo "reference run (no DMTCP) -> $base"
    python3 "$script" baseline "$base" >"$dir/baseline.log" 2>&1 || { tail -20 "$dir/baseline.log"; exit 1; }
fi
args=(-t "${WD:-600}" -d "$dir" -k 'VLLM::EngineCor' -k compile_worker)
[ -n "${EVICT:-}" ] && args+=(-e)
[ -n "${GDB:-}" ] && args+=(-g)
exec "$here/../../scripts/cold_cycle.sh" "${args[@]}" -- python3 "$script" cr "$dir" "$base"
