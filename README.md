# libkfdreplay

**Unprivileged checkpoint/restore of AMD Instinct GPU processes.**

libkfdreplay is a small `LD_PRELOAD` library that lets a ROCm process be snapshotted by
[DMTCP](https://github.com/dmtcp/dmtcp) and restored into a brand-new process — on the same GPU or
on another GPU of the same model — **without root, `CAP_SYS_ADMIN` or `CAP_CHECKPOINT_RESTORE`**.
It works in a stock, unprivileged container (for example a RunPod AMD pod).

On an MI350X a warmed vLLM 0.29 engine (torch.compile + HIP graphs, Qwen3-0.6B, in `sleep(level=2)`)
comes back and answers 12 new prompts **~1.3 s after `dmtcp_restart`**, versus 22–73 s for a normal
start, **token-for-token identical** to an engine that was never snapshotted, with no weights read
from disk and no graphs recaptured.

> **Status: v0.1, experimental.** It has been verified on one GPU model and two ROCm stacks
> (see [What has been verified](#what-has-been-verified)). It mirrors how the ROCm runtime uses the
> kernel driver; a new ROCm release may need an update. The library is **fail-closed**: when it sees
> something it cannot replay, it refuses to checkpoint instead of producing a broken image.

libkfdreplay is the low-level part of AMD checkpoint/restore in
[warmctl](https://github.com/TensorRails/warmctl).

---

## Contents

- [Why](#why)
- [How it works](#how-it-works)
- [What has been verified](#what-has-been-verified)
- [Quick start](#quick-start)
- [API](#api)
- [Using it with vLLM](#using-it-with-vllm)
- [Restore environment requirements](#restore-environment-requirements)
- [Restoring on another GPU](#restoring-on-another-gpu)
- [Fail-closed behaviour](#fail-closed-behaviour)
- [Diagnostics](#diagnostics)
- [Limitations](#limitations)
- [Repository layout and tests](#repository-layout-and-tests)
- [Changes from the research prototype](#changes-from-the-research-prototype)
- [License](#license)

---

## Why

NVIDIA ships `cuda-checkpoint`: the driver moves GPU state into host memory, after which any CPU
checkpointer (CRIU, DMTCP) can save the process. AMD has no equivalent. The official path is CRIU's
`amdgpu` plugin, which drives the `AMDKFD_IOC_CRIU_OP` ioctl. The kernel allows that ioctl only with
`CAP_CHECKPOINT_RESTORE`/`CAP_SYS_ADMIN` in the initial user namespace; in an ordinary container it
fails with **`EACCES`**. Restore into a container that sees only a subset of the host's GPUs has
known problems as well.

libkfdreplay takes another route: it never asks the driver to save anything. It *records* the GPU
objects the runtime creates and *replays* them in the new process through the same unprivileged
ioctls the runtime itself uses.

## How it works

The library sits between the ROCm runtime (ROCr / libhsakmt) and the kernel driver by interposing
`ioctl`, `mmap`/`munmap`, `close`, `open*`, `fopen*` and `access`.

```
 process (under dmtcp_launch, LD_PRELOAD=libkfdreplay.so)
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. record   every KFD object: BOs (VRAM, GTT, userptr,       │
 │             doorbell, MMIO), GPU + CPU mappings, events,     │
 │             AQL queues, process-level settings; DRM syncobjs │
 │             and live GEM/dmabuf/GPU-VA state                 │
 │ 2. kfdr_checkpoint(1)                                        │
 │    a. refuse if anything is not replayable (fail-closed)     │
 │    b. dump VRAM/GTT through ROCr (~11 GB/s on MI350X)        │
 │    c. freeze runtime threads at a barrier                    │
 │    d. replace device mappings with anonymous memory,         │
 │       close /dev/kfd and /dev/dri/renderD*                   │
 └───────────────▲─────────────────────────────────────────────┘
                 │  now a CPU-only process
 3. DMTCP writes the image; the source process exits
 4. dmtcp_restart → new process (new kernel KFD process)
 ┌─────────────────────────────────────────────────────────────┐
 │ 5. kfdr_restore()                                            │
 │    a. reopen the devices on the same fd numbers              │
 │    b. replay ACQUIRE_VM + process-level ioctls, DRM syncobjs │
 │    c. re-create every BO at the same VA with the same handle │
 │       (holes in the handle sequence filled with dummies),    │
 │       write contents back, map it over the old CPU address   │
 │    d. trap handler, scratch; events with the same ids        │
 │    e. queues with the same ids; remap the doorbell page;     │
 │       fast-forward each AQL ring to its old write pointer    │
 │    f. if the GPU changed: translate gpu_id old ↔ new         │
 │    g. release the frozen threads                             │
 └─────────────────────────────────────────────────────────────┘
```

To the runtime nothing happened: every pointer, handle, queue position and event id it holds is
valid again.

Details that make it work in practice (each one found on hardware):

- **Contents dump through ROCr.** `hsa_amd_memory_async_copy` into a locked host buffer while the
  runtime is still alive: 4.25 GB in 0.35 s versus 88 s reading BOs through the PCIe BAR. ROCr is
  reached with `dlopen(RTLD_NOLOAD)` because PyTorch loads HIP with `RTLD_LOCAL`.
- **AQL ring fast-forward.** A new hardware queue starts at `rptr = 0` while ROCr is at `N`. The
  command processor follows *packet headers*, not the doorbell, so the ring is driven through `N`
  no-op barrier packets in laps of the ring size, with all slots outside the current lap marked
  INVALID (~1 M packets/s).
- **Same handles and ids.** The kernel hands out BO handles and event ids in sequence; gaps left by
  freed objects are filled with dummy objects that are released afterwards.
- **ROCm 10 event thread.** ROCr's event thread polls signal memory from user space between waits.
  The freeze waits until every thread that ever waited for events is parked or blocked in a
  syscall; small GTT mappings (signal pools) are swapped atomically for an anonymous copy
  (`mremap`) instead of `PROT_NONE`, and restore fills a BO *before* mapping it over its VA.
- **HIP VMM (vLLM sleep mode) goes through DRM** (dmabuf → GEM handle → `GEM_VA`). That state is
  not replayed: the library requires it to be released at checkpoint time — which is what
  `sleep(level=2)` does — and `wake_up()` rebuilds it in the new process.

## What has been verified

All results: one MI350X (`gfx950`) per process, **unprivileged container on RunPod**, host kernel
`6.12.107+deb13-amd64` with the amdgpu DKMS driver `7.1.3`, DMTCP fork
[`TensorRails/dmtcp` `v5.0.0-warmctl.2`](https://github.com/TensorRails/dmtcp/releases), images read
from the page cache unless stated otherwise. Every cycle is a real cold restart: the source process
exits before `dmtcp_restart` runs.

### Software stacks

| Stack | ROCm userspace | PyTorch | vLLM |
|---|---|---|---|
| A | ROCm 7.0 (`/opt/rocm-7.0.0`), image `rocm/vllm:rocm7.0.0_vllm_0.10.2_20251006` | 2.9.0a0, HIP 7.0.51831 | 0.11.0rc2 (no sleep mode on ROCm) |
| B | ROCm 10.1 **TheRock** (pip wheels), image `rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0` | 2.13.0+rocm10.1.0, HIP 7.16.26385, ROCr 1.21 | 0.29.1.dev0 (sleep mode available) |

### Results

| Workload | Stack | Cycles | Restart → result | Notes |
|---|---|---|---|---|
| HIP toy: non-idempotent kernels, scratch, ring positions `L−1`, `L`, `L+2`, `L+18`, `3L+8` | A | **100/100** | 1.42 s (restart + 1000 kernels + checks) | B: 13/13 |
| PyTorch: bf16 MLP, HIP graph captured once, 2 GiB payload, new GEMM shapes after restore | A | **40/40**, chain 4/4 generations | **0.74 s** to first correct graph result | B: 5/5, 1.1 s |
| vLLM `CuMemAllocator` pool (HIP VMM): sleep → C/R → wake_up | B | 3/3 | `kfdr_restore` 0.09 s | |
| vLLM whole engine, no sleep, Qwen3-0.6B, in-process | A | 10/10 | median 3.2 s (writes 15 GB VRAM) | |
| **vLLM `sleep(2)`, Qwen3-0.6B, in-process** | B | **10/10** | **median 1.26 s** to 12 answers | normal start 22–26 s (warm compile cache), 73.5 s cold |
| **vLLM `sleep(2)`, Qwen3-8B, in-process** | B | **3/3** | **2.1 s** | normal start 56 s |
| **vLLM `sleep(2)`, mp executor** (EngineCore in a child process) | B | **10/10** | median 1.31 s | normal start 50 s |
| **`vllm serve` over HTTP** (worker extension + `/collective_rpc`) | B | **3/3** | **1.29–1.44 s** | normal start 28.5 s |
| mp executor, images evicted from page cache (local NVMe) | B | 3/3 | 1.81–1.87 s | |
| **Restore on another MI350X** (other gpu_id, render node, topology node; same host) | B | **3/3** | 1.0–1.15 s (resume → answers) | |

For vLLM, "correct" means: answers to 12 prompts never seen before the snapshot are
**token-for-token identical** to an independent engine of the same configuration that was never
snapshotted; the repeated batch matches too; the number of weight loads from disk and of graph
captures after restore is **0**.

Negative tests: restoring without writing the buffer contents back (`KFDR_NOCOPY=1`) fails in every
workload (toy, PyTorch, vLLM), as it must.

Breakdown for vLLM `sleep(2)`, Qwen3-0.6B / 8B (stack B):

| Phase | 0.6B | 8B |
|---|---|---|
| Copy weights to CPU (`collective_rpc`) | 1.2 GB, 0.16 s | 16.4 GB, 1.36 s |
| `sleep(2)` frees | 10.2 GiB, 0.03 s | 28.6 GiB, 0.06 s |
| Recorded objects | 4944 BOs, 1650 device mappings, 5 queues | 5356 BOs, 2062 mappings |
| `kfdr_checkpoint` | 2.28 GB dumped, 0.7 s | 2.42 GB, 0.9 s |
| DMTCP image | 8.35 GB | 23.1 GB |
| `dmtcp_restart` → process running | 0.23–0.26 s | 0.35–0.37 s |
| `kfdr_restore` | 0.79–0.92 s | 0.90–0.92 s |
| `wake_up` + reload weights from CPU | 0.095 s | 0.54 s |
| **`dmtcp_restart` → 12 answers** | **1.26–1.41 s** | **2.12–2.14 s** |

## Quick start

### Requirements

- Linux x86-64, AMD Instinct GPU with the amdgpu/KFD driver (verified: MI350X, DKMS 7.1.3).
- `/dev/kfd` and one `/dev/dri/renderD*` visible in the container. No capabilities needed.
- gcc (or clang) and make. **No ROCm SDK is needed to build**: the library declares the few KFD/DRM
  structures it uses itself and links only libc, libdl and libpthread.
- DMTCP: the [TensorRails fork](https://github.com/TensorRails/dmtcp/releases) `v5.0.0-warmctl.2`
  (it adds the `fdkeep` plugin and parallel restart readers used below).

### Build

```bash
make            # -> build/libkfdreplay.so
make test       # GPU-free unit tests
```

### Install DMTCP

```bash
scripts/install_dmtcp.sh /opt/dmtcp
```

```bash
export DMTCP_ROOT=/opt/dmtcp
```

### Run a cycle

A program that uses libkfdreplay looks like this (C/HIP; the library is not linked, its symbols
are found at runtime):

```c
int (*kfdr_checkpoint)(int) = dlsym(RTLD_DEFAULT, "kfdr_checkpoint");
int (*kfdr_restore)(void)   = dlsym(RTLD_DEFAULT, "kfdr_restore");
int (*dmtcp_checkpoint)(void) = dlsym(RTLD_DEFAULT, "dmtcp_checkpoint");

hipDeviceSynchronize();                     // GPU must be idle
if (kfdr_checkpoint(1) != 0) exit(1);       // refused: reason on stderr
if (dmtcp_checkpoint() == 1) _exit(0);      // 1 = image written: source exits
if (kfdr_restore() != 0) exit(1);           // 2 = we are the restored process
/* ... keep using the GPU ... */
```

or from Python:

```python
import ctypes, torch
lib = ctypes.CDLL(None)
torch.cuda.synchronize()
assert lib.kfdr_checkpoint(1) == 0
if lib.dmtcp_checkpoint() == 1:
    os._exit(0)
assert lib.kfdr_restore() == 0
```

`scripts/cold_cycle.sh` runs one full cycle — launch under DMTCP with the library preloaded, check
the image and that the source tree is gone, restart, report — with a watchdog on every phase:

```bash
examples/hip_toy/run.sh 16384 2         # HIP toy, write pointer just past one ring lap
```

```bash
examples/pytorch/run.sh                 # PyTorch + HIP graph
```

```bash
NEG=1 examples/pytorch/run.sh           # negative test: must print RESULT FAIL
```

Output ends with a line such as
`cycle kfdr-pytorch launch_rc=0 restart_rc=0 image=2780MiB launch+checkpoint=6.1s restart+run=17.0s RESULT PASS`
plus the library's summary lines.

`scripts/cold_cycle.sh -h` lists the options: watchdog timeout, extra kill patterns, gdb stacks on
timeout, page-cache eviction for cold-read measurements, chain mode, launch-only / restart-only for
moving images between machines.

### Environment

`scripts/env.sh` sets everything below; the cycle scripts source it.

| Variable | Why |
|---|---|
| `HSA_ENABLE_SDMA=0` | SDMA queues are not replayed; ROCr uses blit kernels instead |
| `HSA_ENABLE_SCRATCH_ASYNC_RECLAIM=0`, `HSA_NO_SCRATCH_RECLAIM=1` | keep scratch memory stable across the checkpoint |
| `OMP_NUM_THREADS=1` (+ OpenBLAS, MKL) | `fork()` with live thread pools hangs the child under DMTCP |
| `USE_LIBUV=0` | torch.distributed's libuv TCPStore accepts sockets behind DMTCP's back; the checkpoint then drains the socket forever |
| `UV_USE_IO_URING=0`, `UCX_MEM_MMAP_HOOK_MODE=none` | bypass DMTCP's wrappers otherwise |
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False` | expandable segments use HIP VMM, i.e. live DRM state (refused) |
| `dmtcp_launch --no-gzip --with-plugin libdmtcp_fdkeep.so` | gzip makes writing a multi-GB image take ~50 s instead of 0.3 s |
| `DMTCP_RESTART_HUGEPAGES=1 DMTCP_RESTART_READERS=16 DMTCP_RESTART_MMAP=0`, `dmtcp_restart --no-strict-checking` | fast parallel image read (fork features) |

Remember that DMTCP restores the process with the environment it had at **launch**. Variables set
for `dmtcp_restart` never reach the restored program; that includes `KFDR_*` flags.

## API

Declared in [`include/kfdreplay.h`](include/kfdreplay.h); all functions log to stderr with the prefix
`[kfdr]`.

| Function | |
|---|---|
| `int kfdr_checkpoint(int cold)` | `cold=1`: dump GPU memory, freeze runtime threads, release the devices; the process can now be saved by DMTCP. `cold=0`: destroy and re-create objects inside the same process (testing aid). Returns 0, or -1 if refused. The GPU must be idle. |
| `int kfdr_restore(void)` | In the restored process, before any other thread uses the GPU: reopen devices, replay everything, release threads. Returns 0 or -1 (the process cannot use the GPU after a failure). |
| `const char *kfdr_last_error(void)` | Reason of the last refusal/failure, `""` after success. |
| `int kfdr_state(void)` | `KFDR_STATE_RUNNING`, `_CHECKPOINTED_WARM`, `_CHECKPOINTED_COLD`, `_RESTORED`. |
| `const char *kfdr_version(void)` | `"0.1.0"`. |
| `void kfdr_drm_report(void)` | Print the DRM ioctls the runtime issued since restore, and DRM errors. |
| `uint64_t kfdr_max_wptr(uint32_t *slots)` | Largest AQL write pointer (testing aid for ring positions). |

Threads other than the caller that enter the driver during a checkpoint block until
`kfdr_restore()` returns.

## Using it with vLLM

The flow is the one warmctl uses on NVIDIA: **copy weights out → `sleep(level=2)` → snapshot →
restore → `wake_up` → copy weights back**. Sleep mode releases the HIP VMM memory pool (weights and
KV cache), so the snapshot holds no live DRM state and only ~2 GB of VRAM remains to dump. It needs
a vLLM with working ROCm sleep mode (verified with 0.29).

Start the engine with the library preloaded under DMTCP and with a worker extension that exposes
the library to the host — [`examples/vllm_serve/kfdr_ext.py`](examples/vllm_serve/kfdr_ext.py):

```bash
. scripts/env.sh
export VLLM_SERVER_DEV_MODE=1 PYTHONPATH=$PWD/examples/vllm_serve
LD_PRELOAD=$KFDR_LIB $DMTCP_ROOT/bin/dmtcp_launch --no-gzip --with-plugin $DMTCP_FDKEEP \
    --new-coordinator -p 7779 --ckptdir /snap \
  vllm serve Qwen/Qwen3-0.6B --enable-sleep-mode --worker-extension-cls kfdr_ext.KfdrExt
```

Checkpoint (every request with `Connection: close` — DMTCP cannot drain a socket whose peer is not
under DMTCP):

```
POST /collective_rpc {"method": "kfdr_export"}       # weights + buffers -> CPU memory of the worker
POST /sleep?level=2                                  # frees weights and KV cache
POST /collective_rpc {"method": "kfdr_checkpoint"}   # -> {"rc": 0, ...}
dmtcp_command -p 7779 --bcheckpoint && dmtcp_command -p 7779 --quit
```

Restore:

```
dmtcp_restart --no-strict-checking --new-coordinator -p 7780 /snap/ckpt_*.dmtcp &
GET  /health                                         # ~0.25 s after dmtcp_restart
POST /collective_rpc {"method": "kfdr_restore"}      # -> {"rc": 0, ...}
POST /wake_up?tags=weights
POST /collective_rpc {"method": "kfdr_import"}       # weights back from the CPU copy
POST /wake_up?tags=kv_cache
```

[`examples/vllm_serve/serve_cr.py`](examples/vllm_serve/serve_cr.py) runs exactly this against a
reference server and compares the answers. Offline engines:
[`examples/vllm_offline/run.sh inprocess`](examples/vllm_offline/) and `run.sh mp` (the library is
inherited by EngineCore through `LD_PRELOAD`; `kfdr_checkpoint`/`kfdr_restore` must run *in the
process that owns the GPU*, i.e. in the worker, via `collective_rpc`).

Notes:

- In the mp executor and in `vllm serve` only EngineCore opens `/dev/kfd`; the API server and the
  frontend do not touch the GPU.
- vLLM 0.29 keeps the CUDA-graph pool in VRAM while asleep (vLLM ≥ 0.32 releases it); it is part of
  the dump.
- Release any `hipHostRegister`ed staging buffers before `sleep()`.

## Restore environment requirements

DMTCP restores memory mappings of files from the files themselves, so the restore side must look
like the checkpoint side:

- **Same files at the same paths, same size and mtime** (nanoseconds — `rsync` keeps only seconds):
  `libkfdreplay.so`, the ROCm/PyTorch/vLLM installation, the model files.
- **JIT caches created on the source**: `~/.triton/cache`, `~/.cache/vllm`, `~/.cache/comgr`,
  `~/.cache/tvm-ffi`, `/tmp/torchinductor_$USER`, `/tmp/vllm_dist_*` (copy them with their mtimes).
- **Same GPU model, firmware and ROCm stack**; one GPU per process.
- HTTP clients talking to a process that will be checkpointed must use `Connection: close`.
- `USE_LIBUV=0` and `OMP_NUM_THREADS=1` at launch (see [Environment](#environment)).

## Restoring on another GPU

If the GPU visible in the restore container differs from the source GPU (verified: another MI350X
of the same host, different `gpu_id`, render node and topology node), the library detects it and
translates transparently:

- `gpu_id` old → new in every ioctl argument that carries it (ALLOC, CREATE_QUEUE, ACQUIRE_VM,
  SET_MEMORY_POLICY, SET_SCRATCH_BACKING_VA, SET_TRAP_HANDLER, GET_CLOCK_COUNTERS, GET_TILE_CONFIG,
  AVAILABLE_MEMORY), in MAP/UNMAP device arrays and in the upper 32 bits of BO handles — both in its
  own replay and in all later runtime calls;
- new → old in values the kernel returns (GET_PROCESS_APERTURES_NEW, GET_DMABUF_INFO);
- the render node is reopened under its new number;
- sysfs topology paths `…/nodes/<old>/…` are rewritten to `…/nodes/<new>/…`.

Translation never modifies the runtime's memory: the MAP/UNMAP device array is one of libhsakmt's
own tables, and rewriting it in place breaks its "all nodes registered" check.

The current GPU is the topology node whose `drm_render_minor` exists in `/dev/dri`; if there is not
exactly one, no translation is attempted.

Moving the snapshot: `scripts/cold_cycle.sh -L -d DIR -- …` on the source (launch and checkpoint
only), copy `DIR` plus the files listed above, then `scripts/cold_cycle.sh -R -d DIR` on the target.

## Fail-closed behaviour

`kfdr_checkpoint()` returns -1 and prints `[kfdr] REFUSE: …` (also in `kfdr_last_error()`) when:

| Condition | Why |
|---|---|
| A successful KFD ioctl the library does not model (e.g. `UPDATE_QUEUE`, `SET_CU_MASK`, `IMPORT_DMABUF`, `SVM`, `DBG_TRAP`, `CRIU_OP`, unknown numbers) | its effect would be missing after restore |
| A successful state-changing DRM ioctl (contexts, command submission, syncobj signal/reset/import/export, …) | render-node state is not replayed |
| Live GEM handles, `GEM_VA` mappings or dmabuf fds at checkpoint time | HIP VMM / dmabuf state; release it first (vLLM: `sleep(level=2)`) |
| A device mapping in `/proc/self/maps` not explained by a recorded BO | something the model does not know about |
| A buffer whose contents cannot be read | the image would be incomplete |
| A thread still inside `WAIT_EVENTS` after 1 s, or an event thread spinning in user space for 5 s | cannot freeze safely |
| A queue with `rptr != wptr` | GPU not idle: synchronize first |

What is replayed, and what is only allowed:

| Driver call | Treatment |
|---|---|
| KFD ALLOC/FREE, MAP/UNMAP, CREATE/DESTROY_EVENT, CREATE/DESTROY_QUEUE | objects: replayed with the same VA/handle/id |
| KFD ACQUIRE_VM, SET_MEMORY_POLICY, SET_SCRATCH_BACKING_VA, SET_TRAP_HANDLER, SET_XNACK_MODE, RUNTIME_ENABLE | last call replayed once |
| KFD queries (GET_VERSION, apertures, clock counters, tile config, available memory, dmabuf info, SET/RESET_EVENT, WAIT_EVENTS, EXPORT_DMABUF) | passed through; the runtime calls them again |
| DRM VERSION, GET_UNIQUE, GET_CAP, AMDGPU_INFO, GEM_MMAP, GEM_METADATA, SYNCOBJ_WAIT/TIMELINE_WAIT/QUERY | read-only, allowed |
| DRM SYNCOBJ_CREATE/DESTROY | journaled and replayed; the handle must match |
| DRM GEM_CREATE, PRIME_FD_TO_HANDLE/HANDLE_TO_FD, GEM_VA, GEM_CLOSE | allowed, but must be released at checkpoint time |

After a refusal the process normally continues unchanged: nothing has been torn down yet and the
frozen threads are released. Two exceptions: a failure in the middle of a *warm* teardown leaves
objects half-destroyed and threads parked, and a failed `kfdr_restore()` leaves the process without
a usable GPU — terminate it in both cases.

## Diagnostics

Environment switches (read at the time of use; remember that a restored process keeps the
environment of `dmtcp_launch`):

| Variable | Effect |
|---|---|
| `KFDR_DEBUG=1` | per-object details: live BOs and mappings at checkpoint, re-created BOs, queue states, fd groups, per-ioctl DRM counts, fast-forward per queue |
| `KFDR_TRACE=1` | after a cold restore, log every KFD ioctl, DRM ioctl and device `mmap` |
| `KFDR_CRASH=1` | on SIGSEGV/SIGBUS/SIGABRT/SIGILL print the address, the recorded mapping it falls into, the thread name and a backtrace |
| `KFDR_DRM_BT=<nr>` | backtrace whenever DRM ioctl number `<nr>` is issued (e.g. `0xbf` = SYNCOBJ_CREATE) |
| `KFDR_DRM_ALLOW=1` | **research only**: downgrade the DRM refusal to a warning |
| `KFDR_NOCOPY=1` | **negative tests only**: do not write buffer contents back on restore; the program must fail |
| `KFDR_COPY=cpu` | dump buffers by reading them through the render node instead of ROCr (slow) |

Marker files on the **restore host** (for switching diagnostics on after the snapshot was taken):

| File | Effect |
|---|---|
| `/tmp/kfdr_trace` | same as `KFDR_TRACE=1` |
| `/tmp/kfdr_pause` | after restore, print the real pid and sleep 40 s so a debugger can attach |

Always printed after a cold restore: every failed KFD ioctl, DRM ioctl, `mmap` of a device and
`open` under `/sys` or `/dev` (first 40 of each). The libhsakmt debug log (`HSAKMT_DEBUG_LEVEL=7`
at launch) together with the trace is the most useful tool for restore problems; gdb on a process
restored by DMTCP is unreliable. `python3 scripts/fingerprint.py` prints the exact runtime stack
for bug reports.

## Limitations

- **One GPU per process; TP > 1 / RCCL is not supported yet.** Multi-GPU processes and
  peer-to-peer mappings have not been tried.
- **Moving between physical hosts is not verified.** Restore on another GPU was tested only between
  two containers on the *same* host (different GPU, render node, topology node).
- **Only MI350X (gfx950) has been tested.** MI300X and other generations are expected to work in
  principle but have not been run.
- **Tied to ROCr versions.** The DRM allow-list and the freeze rules mirror the behaviour of ROCr in
  ROCm 7.0 and 10.1 (TheRock). Each new ROCm release can introduce new driver usage; the fail-closed
  checks will refuse rather than corrupt, but support needs to be re-verified.
- **SDMA queues are not replayed** (`HSA_ENABLE_SDMA=0` is required).
- **Hang on exit after a cross-GPU restore**: after the checks pass, a restored vLLM process may
  print a `FileNotFoundError` from multiprocessing's finalizer (a named semaphore in `/dev/shm`
  created on the source) and once hung on exit until killed. Correctness is not affected; copy
  `/dev/shm` with the image or kill the process after use.
- **Large ring positions.** Fast-forward costs ~1 µs per packet submitted since queue creation; a
  long-running server with ~10⁹ packets would need minutes. Not measured on a long-lived process.
- **Images are large.** The DMTCP image contains the whole CPU memory of the process tree (8.4 GB
  for Qwen3-0.6B, 23 GB for 8B); the numbers above come from the page cache or local NVMe.
  Network storage will dominate restore time.
- **The GPU must be idle and HIP VMM released** at checkpoint time (synchronize; for vLLM, sleep).
- Restore-chain for vLLM (restore → serve → new snapshot) is untested; for PyTorch it works (4/4).
- Linux x86-64 only.

## Repository layout and tests

```
include/kfdreplay.h      public API
src/
  hooks.c                libc interposition, recording of KFD/DRM objects and mappings
  checkpoint.c           refusal checks, contents dump, freeze, device release
  restore.c              reopen, replay, AQL fast-forward
  xlate.c                ioctl classification, gpu_id translation
  topology.c             KFD sysfs topology, path rewriting
  log.c, state.c, api.c  logging/last error, recorded state, small API + crash handler
  kfdr.h                 KFD/DRM ABI subset and internal declarations
examples/                hip_toy, pytorch, vllm_vmm, vllm_offline, vllm_serve (each with run.sh)
scripts/                 env.sh, cold_cycle.sh, install_dmtcp.sh, evict_page_cache.py, fingerprint.py
tests/                   GPU-free tests
```

`make test` runs without a GPU:

- `test_xlate` — ioctl classification and gpu_id / BO-handle translation, including that the
  caller's MAP/UNMAP device array is never modified;
- `test_topology` — topology parsing and path rewriting on a fake sysfs tree;
- `test_abi` — every structure size, field offset, ioctl number and flag in `src/kfdr.h` against the
  kernel UAPI headers (`linux-libc-dev`); `kfd_ioctl_create_queue_args` may be 88 bytes in older
  headers, the library uses the newer 96-byte layout of the amdgpu DKMS driver;
- `test_preload` — the libc hooks are transparent and an empty checkpoint/restore cycle passes;
- `test_exports` — only the API and the interposed libc symbols are exported.

CI builds with gcc and clang on Ubuntu 22.04 and 24.04. The GPU tests are the examples; run them
on a machine with an Instinct GPU.

## Changes from the research prototype

The code comes from the prototype used for all measurements above (`kfdreplay2.c`). The replay
logic is unchanged; the following changes **have not been re-run on hardware yet**:

1. Split into modules, English comments, hidden visibility for internal symbols (the exported
   symbol set is the API plus the interposed libc functions).
2. New API: `kfdr_version()`, `kfdr_state()`, `kfdr_last_error()`; refusal and failure messages are
   also stored for `kfdr_last_error()`.
3. A refusal after threads were frozen now always releases them: "thread still in WAIT_EVENTS" used
   to return with threads parked, "queue busy" cleared the flag without waking them. Nothing has
   been torn down at those points, so the process continues.
4. Diagnostics: `KFDR_TRACE=1` (or `/tmp/kfdr_trace`) now also enables the KFD ioctl trace (the
   prototype enabled it only by the file) and is evaluated once at the end of a cold restore;
   flags are enabled by a value starting with `1` (`KFDR_DEBUG`/`KFDR_TRACE` used to accept any
   value); `KFDR_DUMP_BOS` merged into `KFDR_DEBUG`; per-object lines (fd groups, per-queue state
   and fast-forward, DRM per-ioctl counts, topology rewrites, mappings without a BO, doorbell offset
   changes) moved under `KFDR_DEBUG`.
5. Dead code removed (an `in_hook` flag that was never set, an unused translation counter).

## License

Apache License 2.0 — see [LICENSE](LICENSE). The DMTCP fork used with it is LGPL-3.0 and is not
part of this repository.
