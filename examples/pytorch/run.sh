#!/usr/bin/env bash
# One cold cycle of the PyTorch + HIP graph example.
#   examples/pytorch/run.sh [seed]
#   CHAIN=3 examples/pytorch/run.sh     restore -> checks -> new checkpoint, 3 times
#   NEG=1 examples/pytorch/run.sh       negative test: restore without contents (KFDR_NOCOPY=1) must fail (rc != 0)
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dir=${DIR:-$PWD/kfdr-pytorch}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
args=(-t "${WD:-600}" -d "$dir" -c "${CHAIN:-0}")
[ -n "${NEG:-}" ] && args+=(-E KFDR_NOCOPY=1)
exec "$here/../../scripts/cold_cycle.sh" "${args[@]}" -- python3 "$here/torch_graph.py" "$dir" "${1:-7}"
