/*
 * Recorded state shared by the hooks, checkpoint and restore.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kfdr.h"

#include <dlfcn.h>

struct bo bos[MAX_BO];
int nbo;
struct mp mps[MAX_MAP];
int nmp;
struct q qs[MAX_QUEUE];
int nq;
struct ev evs[MAX_EVENT];
int nev;
const char *record_overflow;
struct pl plv[KFD_NR_MAX];
int pl_order[KFD_NR_MAX], npl;

char fdpath[MAX_DEV_FD][64];
int fdkind[MAX_FD];
int fdlead[MAX_DEV_FD];
/*
 * Device-fd kinds at the time of the cold checkpoint, used only for reopen.
 * fdkind[] itself is reset when an fd is closed: the number of the closed
 * /dev/kfd is reused immediately (DMTCP opens /proc/self/pagemap and issues
 * PAGEMAP_SCAN on it), and a stale "KFD" kind would park DMTCP's image-writer
 * thread at the freeze barrier forever.
 */
int cold_kind[MAX_DEV_FD];
int kfd_fd = -1, render_fd = -1;
uint64_t signal_handle;

unsigned long unknown_calls;
unsigned unknown_nr;
unsigned long partial_overlaps;

/* Inventory of DRM ioctls on the render node (not replayed), by _IOC_NR. */
unsigned long drm_ok[256], drm_err[256], drm_ok_since_restore[256];
/*
 * DRM syncobjs (ROCm 10.x ROCr creates one at start-up): CREATE/DESTROY are
 * journaled and replayed after a cold restore, checking that the kernel hands
 * out the same handle. Any other syncobj operation changes state that is not
 * replayed and makes the checkpoint refuse.
 */
struct syncobj_rec sobj[MAX_SYNCOBJ];
int nsobj;
/*
 * Live DRM state (HIP VMM path in ROCm 10.x: KFD BO -> dmabuf -> GEM handle ->
 * GEM_VA at a reserved VA). A checkpoint is allowed only when all of it has
 * been released: no live GEM handles, GEM_VA mappings or dmabuf fds.
 */
/* A bit per live handle: a large model under vLLM's sleep-mode allocator holds
 * thousands at once, and releases them all before the checkpoint. */
uint64_t gem_live[MAX_GEM_HANDLE / 64];
int ngem_live;
long gem_va_live;
unsigned long drm_track_overflow;

pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cv = PTHREAD_COND_INITIALIZER;
volatile int frozen, inflight_wait, parked;
long waiter_tid[MAX_WAITER]; /* threads that have ever entered WAIT_EVENTS */
int nwaiter;
pthread_t controller;

int cold_state;
int restored_cold;

int (*real_ioctl)(int, unsigned long, ...);
void *(*real_mmap)(void *, size_t, int, int, int, off_t);
int (*real_munmap)(void *, size_t);
int (*real_close)(int);

void kfdr_resolve_real(void) {
    if (real_ioctl) return;
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    real_mmap = dlsym(RTLD_NEXT, "mmap");
    real_munmap = dlsym(RTLD_NEXT, "munmap");
    real_close = dlsym(RTLD_NEXT, "close");
}
