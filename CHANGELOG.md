# Changelog

## Unreleased (0.1.0-dev)

Initial standalone repository ported from the `kfdreplay2` research prototype.

### Library (`libkfdreplay.so`)

- Split the prototype into modules (`hooks`, `checkpoint`, `restore`, `xlate`,
  `topology`, `log`, `state`, `api`) with hidden symbol visibility; only the
  public `kfdr_*` API and the interposed libc functions are exported.
- Added `kfdr_version()`, `kfdr_state()` and `kfdr_last_error()`.
- Fixed thread unfreezing on early checkpoint refusals (`WAIT_EVENTS` timeout
  and busy AQL queue) so a refused checkpoint leaves the process usable.
- Unified diagnostic switches (`KFDR_DEBUG=1`, `KFDR_TRACE=1`, `KFDR_CRASH=1`,
  `/tmp/kfdr_trace`, `/tmp/kfdr_pause`).

### Cold-cycle harness, examples and tests

- Added `scripts/cold_cycle.sh`, `scripts/env.sh`, `scripts/install_dmtcp.sh`,
  `scripts/evict_page_cache.py` and `scripts/fingerprint.py`.
- Hardened `scripts/cold_cycle.sh` against real DMTCP runs (verified on x86-64
  Linux with `TensorRails/dmtcp` `v5.0.0-warmctl.2`):
  - protected the caller's ancestor chain and subshells from `kill_tree()` and
    orphan checks so `cold_cycle.sh` and outer wrappers (`examples/*/run.sh`)
    never match or kill themselves;
  - tracked per-cycle processes via coordinator `--list`, isolated
    `DMTCP_TMPDIR` and `KFDR_CYCLE_ID` across all chain ports (`PORT`,
    `PORT + 1 + 2*i`), including unprivileged containers where `PR_SET_MM`
    cannot restore `/proc/<pid>/cmdline`;
  - closed stdin (`</dev/null`) on DMTCP/gdb invocations and reaped the
    watchdog's `sleep` child on `unwatch` so piped scripts and `$()` command
    substitutions neither lose stdin nor hang waiting on open descriptors;
  - replaced `bc` arithmetic with `awk`.
- Fixed `examples/vllm_offline/run.sh`, `examples/vllm_serve/run.sh` and
  `examples/vllm_serve/serve_cr.py` so reference `baseline.json` runs are not
  forced into `HF_HUB_OFFLINE=1` before model weights are cached, and added
  `NEG=1` pass-through to `examples/vllm_offline/run.sh`.
- Added GPU-free unit tests (`make test`) and a real-DMTCP harness integration
  test (`make test-dmtcp` / `tests/test_cold_cycle.sh`) wired into CI.
