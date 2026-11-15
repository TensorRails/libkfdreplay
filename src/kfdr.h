/*
 * libkfdreplay internal header: KFD/DRM ABI subset, recorded state, helpers.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * The library deliberately does not include ROCm or libdrm headers: the few
 * KFD/DRM structures it needs are declared here, mirroring
 * <linux/kfd_ioctl.h> and <drm/drm.h>/<drm/amdgpu_drm.h>. tests/test_abi.c
 * checks every size, ioctl number and field offset against the kernel UAPI
 * headers when they are installed.
 */
#ifndef KFDR_INTERNAL_H
#define KFDR_INTERNAL_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>

#include "../include/kfdreplay.h"

/* Public (exported) symbols; everything else is built with hidden visibility. */
#define KFDR_API __attribute__((visibility("default")))

/* ---- KFD ioctl ABI (subset of <linux/kfd_ioctl.h>) ------------------------ */

/* kfd_ioctl_alloc_memory_of_gpu_args */
struct alloc_args { uint64_t va_addr, size, handle, mmap_offset; uint32_t gpu_id, flags; };
/* kfd_ioctl_free_memory_of_gpu_args */
struct free_args { uint64_t handle; };
/* kfd_ioctl_map_memory_to_gpu_args / kfd_ioctl_unmap_memory_from_gpu_args */
struct map_args { uint64_t handle, dev_ptr; uint32_t n_devices, n_success; };
/* kfd_ioctl_create_queue_args */
struct cq_args {
    uint64_t ring, wptr, rptr, doorbell_offset;
    uint32_t ring_size, gpu_id, queue_type, pct, prio, queue_id;
    uint64_t eop, eop_size, ctx_save;
    uint32_t ctx_save_size, ctl_stack_size, sdma_engine_id, pad;
};
/* kfd_ioctl_destroy_queue_args */
struct dq_args { uint32_t queue_id, pad; };
/* kfd_ioctl_create_event_args */
struct ce_args { uint64_t page_offset; uint32_t trigger, type, auto_reset, node_id, event_id, slot; };
/* kfd_ioctl_destroy_event_args / kfd_ioctl_set_event_args */
struct ev_args { uint32_t event_id, pad; };

#define KFD_IOC(dir, nr, T) _IOC(dir, 'K', nr, sizeof(T))
#define IOC_CREATE_QUEUE  KFD_IOC(_IOC_READ | _IOC_WRITE, 0x02, struct cq_args)
#define IOC_DESTROY_QUEUE KFD_IOC(_IOC_READ | _IOC_WRITE, 0x03, struct dq_args)
#define IOC_CREATE_EVENT  KFD_IOC(_IOC_READ | _IOC_WRITE, 0x08, struct ce_args)
#define IOC_DESTROY_EVENT KFD_IOC(_IOC_WRITE, 0x09, struct ev_args)
#define IOC_SET_EVENT     KFD_IOC(_IOC_WRITE, 0x0A, struct ev_args)
#define IOC_ALLOC         KFD_IOC(_IOC_READ | _IOC_WRITE, 0x16, struct alloc_args)
#define IOC_FREE          KFD_IOC(_IOC_WRITE, 0x17, struct free_args)
#define IOC_MAP           KFD_IOC(_IOC_READ | _IOC_WRITE, 0x18, struct map_args)
#define IOC_UNMAP         KFD_IOC(_IOC_READ | _IOC_WRITE, 0x19, struct map_args)

/* KFD ioctl numbers (_IOC_NR) referenced by name. */
enum {
    NR_GET_VERSION = 0x01,
    NR_CREATE_QUEUE = 0x02,
    NR_DESTROY_QUEUE = 0x03,
    NR_SET_MEMORY_POLICY = 0x04,
    NR_GET_CLOCK_COUNTERS = 0x05,
    NR_CREATE_EVENT = 0x08,
    NR_DESTROY_EVENT = 0x09,
    NR_WAIT_EVENTS = 0x0C,
    NR_SET_SCRATCH_BACKING_VA = 0x11,
    NR_GET_TILE_CONFIG = 0x12,
    NR_SET_TRAP_HANDLER = 0x13,
    NR_GET_PROCESS_APERTURES_NEW = 0x14,
    NR_ACQUIRE_VM = 0x15,
    NR_ALLOC_MEMORY_OF_GPU = 0x16,
    NR_FREE_MEMORY_OF_GPU = 0x17,
    NR_MAP_MEMORY_TO_GPU = 0x18,
    NR_UNMAP_MEMORY_FROM_GPU = 0x19,
    NR_GET_DMABUF_INFO = 0x1C,
    NR_AVAILABLE_MEMORY = 0x23,
    NR_EXPORT_DMABUF = 0x24,
};
#define KFD_NR_MAX 0x30

/* KFD_IOC_ALLOC_MEM_FLAGS_* */
#define F_VRAM 1u
#define F_GTT 2u
#define F_USERPTR 4u
#define F_DOORBELL 8u
#define F_MMIO 16u
#define F_WRITABLE (1u << 31)

/* AQL queue type (KFD_IOC_QUEUE_TYPE_COMPUTE_AQL) and packet header types. */
#define QUEUE_TYPE_COMPUTE_AQL 2
#define AQL_PACKET_SIZE 64
#define AQL_HDR_INVALID 1
#define AQL_HDR_BARRIER_AND 3

/*
 * KFD ioctl classes, indexed by _IOC_NR (see kfd_ioctl_class[] in xlate.c):
 *   'P'  process-level setting: the last successful call is kept and replayed
 *        once after a cold restore, before any object is re-created;
 *   'R'  query / stateless: passed through, never replayed;
 *   'S'  object create/destroy: tracked individually (BOs, queues, events);
 *   0    unknown or unsupported: any successful call makes kfdr_checkpoint()
 *        refuse.
 */
extern const char kfd_ioctl_class[KFD_NR_MAX];
char kfdr_classify_kfd(unsigned nr);

/* ---- DRM ioctl numbers (subset of <drm/drm.h>, <drm/amdgpu_drm.h>) --------- */
#define DRM_NR_VERSION 0x00
#define DRM_NR_GET_UNIQUE 0x01
#define DRM_NR_GEM_CLOSE 0x09
#define DRM_NR_GET_CAP 0x0c
#define DRM_NR_PRIME_H2FD 0x2d
#define DRM_NR_PRIME_FD2H 0x2e
#define DRM_NR_AMDGPU_GEM_CREATE 0x40
#define DRM_NR_AMDGPU_GEM_MMAP 0x41
#define DRM_NR_AMDGPU_INFO 0x45
#define DRM_NR_AMDGPU_GEM_METADATA 0x46
#define DRM_NR_AMDGPU_GEM_VA 0x48
#define DRM_NR_SYNCOBJ_CREATE 0xbf
#define DRM_NR_SYNCOBJ_DESTROY 0xc0
#define DRM_NR_SYNCOBJ_WAIT 0xc3
#define DRM_NR_SYNCOBJ_TIMELINE_WAIT 0xca
#define DRM_NR_SYNCOBJ_QUERY 0xcb
#define AMDGPU_VA_OP_MAP 1
#define AMDGPU_VA_OP_UNMAP 2

/* 1 if a successful DRM ioctl with this number leaves no state that a cold
   restore would have to replay (or the state is tracked separately). */
int kfdr_drm_nr_allowed(unsigned nr);

/* ---- recorded state -------------------------------------------------------- */

#define MAX_BO 32768
#define MAX_MAP 32768
#define MAX_QUEUE 256
#define MAX_EVENT 8192
#define MAX_SYNCOBJ 64
#define MAX_GEM_HANDLE (1u << 20) /* GEM handles are small idr ids; one bit each */
#define MAX_WAITER 32
#define MAX_FD 4096     /* fd numbers tracked by fdkind[] */
#define MAX_DEV_FD 1024 /* fd numbers that may be device fds (reopened on restore) */

/* A KFD buffer object (ALLOC_MEMORY_OF_GPU). */
struct bo {
    struct alloc_args a;  /* output args of ALLOC (after restore: new handle/offset) */
    uint32_t gpus[16], ngpu; /* GPUs it is mapped to (MAP_MEMORY_TO_GPU) */
    int live;
    void *data;           /* host copy of the contents between checkpoint and restore */
    uint64_t new_off;     /* mmap offset of the re-created BO */
    uint64_t user_addr;   /* userptr BOs: the user address passed to ALLOC */
};

/* A CPU mapping of /dev/kfd or a render node. */
struct mp {
    void *addr;
    size_t len;
    int prot, flags, fd;
    uint64_t off;
    int live;
    struct bo *owner; /* BO the mapping belongs to (resolved at restore) */
    int snap;         /* cold: replaced by an anonymous RW copy (small GTT) */
};

struct q { struct cq_args a; int live; };
struct ev { struct ce_args a; int live, page; };
/* Last argument block of a process-level ('P') ioctl. */
struct pl { unsigned long req; unsigned char arg[128]; int set; };
/* DRM syncobj create/destroy journal entry, replayed in order. */
struct syncobj_rec { int fd; uint32_t handle, flags, destroy; };

/* fdkind[] values. */
enum { FD_UNCLASSIFIED = 0, FD_KFD = 1, FD_RENDER = 2, FD_OTHER = 3 };

extern struct bo bos[MAX_BO];
extern int nbo;
extern struct mp mps[MAX_MAP];
extern int nmp;
extern struct q qs[MAX_QUEUE];
extern int nq;
extern struct ev evs[MAX_EVENT];
extern int nev;
extern struct pl plv[KFD_NR_MAX];
extern int pl_order[KFD_NR_MAX], npl;

extern char fdpath[MAX_DEV_FD][64];
extern int fdkind[MAX_FD];
extern int fdlead[MAX_DEV_FD];   /* lowest fd sharing the same open file (kcmp) */
extern int cold_kind[MAX_DEV_FD]; /* device-fd kinds at cold checkpoint, for reopen */
extern int kfd_fd, render_fd;
extern uint64_t signal_handle;

extern unsigned long unknown_calls;
extern unsigned unknown_nr;
extern unsigned long partial_overlaps;

extern unsigned long drm_ok[256], drm_err[256], drm_ok_since_restore[256];
extern struct syncobj_rec sobj[MAX_SYNCOBJ];
extern int nsobj;
extern uint64_t gem_live[MAX_GEM_HANDLE / 64];
extern int ngem_live;
extern long gem_va_live;
extern unsigned long drm_track_overflow;

extern pthread_mutex_t mu;
extern pthread_cond_t cv;
extern volatile int frozen, inflight_wait, parked;
extern long waiter_tid[MAX_WAITER];
extern int nwaiter;
extern pthread_t controller;

extern int cold_state;    /* 1 between a cold checkpoint and the following restore */
extern int restored_cold; /* 1 once a cold restore completed in this process */

/* Real libc entry points (resolved with RTLD_NEXT). */
extern int (*real_ioctl)(int, unsigned long, ...);
extern void *(*real_mmap)(void *, size_t, int, int, int, off_t);
extern int (*real_munmap)(void *, size_t);
extern int (*real_close)(int);
void kfdr_resolve_real(void);

/* ---- hooks.c ---------------------------------------------------------------- */
int kfdr_kind_of(int fd);
int kfdr_maps_bo(const struct mp *p, const struct bo *b);
struct bo *kfdr_bo_by_handle(uint64_t h);

/* ---- checkpoint.c ------------------------------------------------------------ */
int kfdr_kio(unsigned long req, void *a, const char *what);

/* ---- xlate.c: gpu_id translation when restoring on another GPU ------------ */
extern uint32_t gpu_old, gpu_new;
struct xundo {
    uint32_t *g; uint32_t gv;
    uint64_t *h; uint64_t hv;
    uint64_t *dp; uint64_t dpv;
};
uint32_t kfdr_tg(uint32_t g);
uint64_t kfdr_th(uint64_t h);
uint32_t kfdr_tg_back(uint32_t g);
void kfdr_xlate(unsigned nr, void *arg, struct xundo *u);
void kfdr_xundo_apply(struct xundo *u);
void kfdr_xlate_out(unsigned nr, void *arg);

/* ---- topology.c ------------------------------------------------------------ */
extern int topo_old, topo_new;
extern const char *kfdr_topology_dir; /* default /sys/class/kfd/kfd/topology/nodes */
extern const char *kfdr_dri_dir;      /* default /dev/dri */
const char *kfdr_topo_xlate(const char *p, char *buf, size_t n);
int kfdr_topo_node_of(uint32_t gpu_id);
uint32_t kfdr_sysfs_gpu_id(void);

/* ---- log.c ----------------------------------------------------------------- */
double kfdr_now(void);
int kfdr_env_flag(const char *name); /* 1 if the variable starts with '1' */
int kfdr_debug_on(void);             /* KFDR_DEBUG=1 */
int kfdr_trace_on(void);             /* after a cold restore: KFDR_TRACE=1 or /tmp/kfdr_trace exists */
void kfdr_trace_refresh(void);
void kfdr_clear_error(void);
/* Log "[kfdr] <msg>" to stderr and remember it for kfdr_last_error(). */
void kfdr_fail(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

#define KFDR_TRACE_FILE "/tmp/kfdr_trace"
#define KFDR_PAUSE_FILE "/tmp/kfdr_pause"

#define kfdr_log(...) fprintf(stderr, "[kfdr] " __VA_ARGS__)
#define kfdr_dbg(...) do { if (kfdr_debug_on()) fprintf(stderr, "[kfdr] " __VA_ARGS__); } while (0)

#endif /* KFDR_INTERNAL_H */
