# shellcheck shell=bash
# Environment for running a ROCm process under DMTCP + libkfdreplay.
# Source it:  . scripts/env.sh
#
# Inputs (all optional):
#   KFDR_LIB     path to libkfdreplay.so      (default: <repo>/build/libkfdreplay.so)
#   DMTCP_ROOT   DMTCP install prefix with bin/dmtcp_launch
#                (default: /opt/dmtcp, then a dmtcp_launch found in PATH)
#
# Everything exported here is part of the *launch* environment. DMTCP restores
# the process with the environment it had at dmtcp_launch: variables set only
# for dmtcp_restart never reach the restored process.

_kfdr_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export KFDR_LIB=${KFDR_LIB:-$_kfdr_repo/build/libkfdreplay.so}

if [ -z "${DMTCP_ROOT:-}" ]; then
    if [ -x /opt/dmtcp/bin/dmtcp_launch ]; then
        DMTCP_ROOT=/opt/dmtcp
    elif command -v dmtcp_launch >/dev/null 2>&1; then
        DMTCP_ROOT=$(cd "$(dirname "$(command -v dmtcp_launch)")/.." && pwd)
    fi
fi
export DMTCP_ROOT
export DMTCP_BIN=${DMTCP_ROOT:+$DMTCP_ROOT/bin}
# fdkeep plugin of the TensorRails DMTCP fork (location differs between builds).
if [ -n "${DMTCP_ROOT:-}" ] && [ -z "${DMTCP_FDKEEP:-}" ]; then
    for p in "$DMTCP_ROOT/lib/dmtcp/libdmtcp_fdkeep.so" "$DMTCP_ROOT/lib/libdmtcp_fdkeep.so"; do
        [ -f "$p" ] && DMTCP_FDKEEP=$p && break
    done
fi
export DMTCP_FDKEEP

# --- ROCm runtime --------------------------------------------------------------
# SDMA queues are not replayed (no fast-forward for ROCr's SDMA ring protocol):
# make ROCr use blit kernels instead.
export HSA_ENABLE_SDMA=0
# Keep scratch memory stable between checkpoint and restore.
export HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0 HSA_NO_SCRATCH_RECLAIM=1

# --- DMTCP-friendly process environment (same as warmctl) ----------------------
# fork() while OpenMP/OpenBLAS thread pools are alive hangs the child under DMTCP.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# io_uring and UCX mmap hooks bypass DMTCP's wrappers.
export UV_USE_IO_URING=0 UCX_MEM_MMAP_HOOK_MODE=none
# torch.distributed TCPStore on libuv accepts connections behind DMTCP's back;
# the checkpoint then waits forever to drain the socket. Use the classic backend.
export USE_LIBUV=0
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

# --- dmtcp_restart -------------------------------------------------------------
# Restart-side tuning of the TensorRails DMTCP fork (parallel image readers,
# huge pages, read instead of mmap). Only dmtcp_restart reads these.
export KFDR_RESTART_ENV="DMTCP_RESTART_HUGEPAGES=1 DMTCP_RESTART_READERS=16 DMTCP_RESTART_MMAP=0"

kfdr_check_env() {
    local ok=0
    [ -f "$KFDR_LIB" ] || { echo "libkfdreplay not found: $KFDR_LIB (run make, or set KFDR_LIB)" >&2; ok=1; }
    [ -n "${DMTCP_BIN:-}" ] && [ -x "$DMTCP_BIN/dmtcp_launch" ] ||
        { echo "DMTCP not found: set DMTCP_ROOT (see scripts/install_dmtcp.sh)" >&2; ok=1; }
    [ -n "${DMTCP_FDKEEP:-}" ] || echo "warning: DMTCP fdkeep plugin not found; launching without it" >&2
    return $ok
}
