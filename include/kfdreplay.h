/*
 * libkfdreplay — unprivileged checkpoint/restore of AMD GPU (ROCm/KFD) processes.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The library is loaded with LD_PRELOAD into the process that uses the GPU.
 * From process start it records every KFD (/dev/kfd) and relevant DRM
 * (/dev/dri/renderD*) object the ROCm runtime creates. The process itself then
 * calls:
 *
 *   kfdr_checkpoint(1)   dump GPU memory, freeze runtime threads, release the
 *                        device files; the process is now CPU-only and can be
 *                        saved by DMTCP (dmtcp_checkpoint / dmtcp_command).
 *   kfdr_restore()       in the restored process (after dmtcp_restart): reopen
 *                        the devices and replay every object at the same
 *                        virtual addresses and handles, write the contents back,
 *                        unfreeze.
 *
 * Every symbol below can also be looked up at runtime with
 * dlsym(RTLD_DEFAULT, ...) or, from Python, ctypes.CDLL(None).<name>, which is
 * how the examples use it: the application does not link against the library.
 *
 * All functions log to stderr with the prefix "[kfdr]".
 * This is experimental software; see README.md for what has been verified.
 */
#ifndef KFDREPLAY_H
#define KFDREPLAY_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KFDR_VERSION_MAJOR 0
#define KFDR_VERSION_MINOR 1
#define KFDR_VERSION_PATCH 1
#define KFDR_VERSION_STRING "0.1.1"

/* Values returned by kfdr_state(). */
enum kfdr_state {
    /* Normal operation; no checkpoint outstanding (or never restored). */
    KFDR_STATE_RUNNING = 0,
    /* After kfdr_checkpoint(0): GPU objects destroyed in this process,
       runtime threads parked; kfdr_restore() re-creates them in place. */
    KFDR_STATE_CHECKPOINTED_WARM = 1,
    /* After kfdr_checkpoint(1): device files closed, device mappings
       replaced; the process may be saved by DMTCP. */
    KFDR_STATE_CHECKPOINTED_COLD = 2,
    /* Running in a process restored from a cold checkpoint. */
    KFDR_STATE_RESTORED = 3,
};

/*
 * Checkpoint the GPU state of the calling process.
 *
 *   cold = 1  "cold" checkpoint for a process snapshot (DMTCP). GPU memory is
 *             copied to host memory, runtime threads that touch the driver are
 *             parked, device mappings are replaced by anonymous memory and the
 *             /dev/kfd and render-node file descriptors are closed. Call
 *             dmtcp_checkpoint() (or let an external dmtcp_command take the
 *             checkpoint) right after this returns 0, and kfdr_restore() in the
 *             restored process.
 *   cold = 0  "warm" checkpoint inside the same process (testing aid): queues,
 *             events and buffers are destroyed; kfdr_restore() re-creates them.
 *
 * Preconditions: the GPU must be idle (all queues drained, e.g. after
 * hipDeviceSynchronize()/torch.cuda.synchronize()); for vLLM, the engine must
 * be asleep (sleep(level=2)) so that no HIP VMM / DRM GEM state is live.
 *
 * Returns 0 on success, -1 if the checkpoint is refused. Refusal is
 * fail-closed: the library refuses whenever it sees state it cannot replay
 * (an unknown KFD ioctl, a state-changing DRM ioctl, a device mapping that is
 * not explained by a recorded buffer, a live GEM handle / GPU VA mapping /
 * dmabuf fd, a busy queue, a runtime thread that will not settle, or a buffer
 * whose contents cannot be read). The reason is printed to stderr and returned
 * by kfdr_last_error(). On a refusal the process normally keeps running
 * unchanged; see README.md ("Fail-closed behaviour") for the exceptions.
 *
 * Must be called from one thread; other threads that enter the driver while
 * the checkpoint is in progress block until kfdr_restore() returns.
 */
int kfdr_checkpoint(int cold);

/*
 * Restore the GPU state recorded by the last successful kfdr_checkpoint().
 *
 * After a cold checkpoint this must run in the restored process (after
 * dmtcp_restart) before any other thread uses the GPU. It reopens the device
 * files on the same descriptor numbers, replays process-level settings, buffer
 * objects (same VA, same handles), events and queues, writes the saved
 * contents back and finally releases the parked threads.
 *
 * If the process is restored on a different GPU of the same model (another
 * gpu_id / render node), gpu_id values are translated transparently; see
 * README.md ("Restoring on another GPU").
 *
 * Returns 0 on success, -1 on failure (reason on stderr and in
 * kfdr_last_error()). After a failed cold restore the process cannot use the
 * GPU and should be terminated.
 */
int kfdr_restore(void);

/*
 * Print to stderr the DRM ioctls the runtime issued on the render node since
 * the last cold restore, plus all DRM ioctl errors. Diagnostic aid.
 */
void kfdr_drm_report(void);

/*
 * Largest write pointer over all live AQL queues of the process; if slots is
 * not NULL it receives the ring size (in 64-byte packets) of that queue.
 * Testing aid used to drive a queue to ring-boundary positions.
 */
uint64_t kfdr_max_wptr(uint32_t *slots);

/* Library version, e.g. "0.1.0". */
const char *kfdr_version(void);

/* Current state, one of enum kfdr_state. */
int kfdr_state(void);

/*
 * Human-readable reason of the last refused checkpoint or failed restore, or
 * an empty string if the last kfdr_checkpoint()/kfdr_restore() succeeded.
 * The returned pointer stays valid until the next call into the library.
 */
const char *kfdr_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* KFDREPLAY_H */
