#!/usr/bin/env bash
# `vllm serve` over HTTP through a cold cycle (serve_cr.py orchestrates DMTCP itself).
#   examples/vllm_serve/run.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=${DIR:-$PWD/kfdr-serve}
mkdir -p "$dir"
. "$here/../../scripts/env.sh"
kfdr_check_env
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
[ -f "$dir/baseline.json" ] || python3 "$here/serve_cr.py" baseline "$dir/base"
[ -f "$dir/baseline.json" ] || cp "$dir/base/baseline.json" "$dir/baseline.json"
exec python3 "$here/serve_cr.py" cr "$dir/cr" "$dir/baseline.json"
