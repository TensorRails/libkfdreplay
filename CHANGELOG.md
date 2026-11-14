# Changelog

## v0.1.0 — 2026-11-15

Initial public release of `libkfdreplay`, ported from the `kfdreplay2` research prototype.

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
  forced into `HF_HUB_OFFLINE=1` before model weights are cached, added `NEG=1`
  pass-through to `examples/vllm_offline/run.sh`, and reaped the `EngineCore`
  child in `examples/vllm_offline/vllm_mp.py` before source exit.
- Cleaned up DMTCP `ckpt_*_files/` directories and restart scripts alongside
  `ckpt_*.dmtcp` when `-K` is not set.
- Added GPU-free unit tests (`make test`) and a real-DMTCP harness integration
  test (`make test-dmtcp` / `tests/test_cold_cycle.sh`) wired into CI.
- Re-verified the modular library and `scripts/cold_cycle.sh` on MI350X
  (ROCm 10.1 TheRock, PyTorch 2.13, vLLM 0.29, `TensorRails/dmtcp`
  `v5.0.0-warmctl.2`): HIP toy (5 ring boundary positions + negative test),
  PyTorch + HIP graph (3 cycles, 4-generation chain + negative test), vLLM VMM
  pool, vLLM `sleep(2)` in-process (3/3 + negative test) and mp executor (3/3
  including cold page-cache eviction), and `vllm serve` over HTTP.
