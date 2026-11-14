#!/usr/bin/env bash
# One cold cycle of the vLLM CuMemAllocator (HIP VMM) pool example. Needs vLLM >= 0.25 with ROCm sleep mode.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=${DIR:-$PWD/kfdr-vmm}
exec "$here/../../scripts/cold_cycle.sh" -t "${WD:-300}" -d "$dir" -- python3 "$here/vmm_pool.py" "$dir"
