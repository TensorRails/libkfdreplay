/*
 * kfdr_checkpoint(): refuse unless replayable, dump contents, freeze, release devices.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kfdr.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#ifndef KCMP_FILE
#define KCMP_FILE 0
#endif

/* ---- KFD helper used by checkpoint and restore ------------------------------ */

/*
 * Issue a KFD ioctl from the library itself (replay or teardown), with gpu_id
 * translation. Only the device-array pointer is given back afterwards; gpu_ids
 * in our own argument blocks are left as the kernel returned them.
 */
int kfdr_kio(unsigned long req, void *a, const char *what) {
    struct xundo u;
    kfdr_xlate(_IOC_NR(req), a, &u);
    int r = real_ioctl(kfd_fd, req, a);
    if (u.dp) *u.dp = u.dpv;
    if (r) kfdr_fail("%s failed: %s", what, strerror(errno));
    return r;
}

/* ---- contents dump ----------------------------------------------------------- */
/*
 * hsa_memory_copy is not usable: for a VA ROCr does not know it falls back to
 * a CPU memmove. Use hsa_amd_memory_async_copy with explicit agents
 * (GPU -> CPU) into a locked host buffer, while the runtime is still alive.
 * ROCr is reached through dlopen(RTLD_NOLOAD) because PyTorch loads HIP with
 * RTLD_LOCAL and dlsym(RTLD_DEFAULT) would not find it. Fallback: CPU read of
 * the BO through the render node (slow: ~50 s for 2.5 GB vs 0.2 s).
 */
typedef struct { uint64_t handle; } hsa_agent_t;
typedef struct { uint64_t handle; } hsa_signal_t;
#define HSA_AGENT_INFO_DEVICE 17
#define HSA_DEVICE_TYPE_CPU 0
#define HSA_DEVICE_TYPE_GPU 1
#define HSA_SIGNAL_CONDITION_LT 2
#define HSA_WAIT_STATE_BLOCKED 0

static int (*h_lock)(void *, size_t, hsa_agent_t *, int, void **);
static int (*h_unlock)(void *);
static int (*h_acopy)(void *, hsa_agent_t, const void *, hsa_agent_t, size_t, uint32_t, const hsa_signal_t *, hsa_signal_t);
static int (*h_iter)(int (*)(hsa_agent_t, void *), void *);
static int (*h_info)(hsa_agent_t, int, void *);
static int (*h_screate)(int64_t, uint32_t, const hsa_agent_t *, hsa_signal_t *);
static int (*h_sdestroy)(hsa_signal_t);
static int64_t (*h_swait)(hsa_signal_t, int, int64_t, uint64_t, int);
static void (*h_sstore)(hsa_signal_t, int64_t);
static hsa_agent_t gpu_agent, cpu_agent;
static int have_gpu, have_cpu;

static int agent_cb(hsa_agent_t a, void *u) {
    (void)u;
    int type = -1;
    h_info(a, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU && !have_gpu) { gpu_agent = a; have_gpu = 1; }
    if (type == HSA_DEVICE_TYPE_CPU && !have_cpu) { cpu_agent = a; have_cpu = 1; }
    return 0;
}

static const char *copy_mode = "cpu";

/* KFDR_COPY=cpu forces the CPU read path. */
static void dump_contents(int *nlost, double *bytes, int cold) {
    const char *mode = getenv("KFDR_COPY");
    int use_hsa = !(mode && !strcmp(mode, "cpu"));
    void *hh = dlopen("libhsa-runtime64.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!hh) hh = RTLD_DEFAULT;
#define SYM(v, n) do { *(void **)&v = dlsym(hh, n); if (!v) use_hsa = 0; } while (0)
    if (use_hsa) {
        SYM(h_lock, "hsa_amd_memory_lock");
        SYM(h_unlock, "hsa_amd_memory_unlock");
        SYM(h_acopy, "hsa_amd_memory_async_copy");
        SYM(h_iter, "hsa_iterate_agents");
        SYM(h_info, "hsa_agent_get_info");
        SYM(h_screate, "hsa_signal_create");
        SYM(h_sdestroy, "hsa_signal_destroy");
        SYM(h_swait, "hsa_signal_wait_scacquire");
        SYM(h_sstore, "hsa_signal_store_relaxed");
    }
#undef SYM
    hsa_signal_t sig = {0};
    if (use_hsa) {
        h_iter(agent_cb, NULL);
        use_hsa = have_gpu && have_cpu && !h_screate(1, 0, NULL, &sig);
    }
    copy_mode = use_hsa ? "hsa_amd_memory_async_copy" : "cpu";
    for (int i = 0; i < nbo; i++) {
        struct bo *b = &bos[i];
        if (!b->live || !(b->a.flags & (F_VRAM | F_GTT))) continue;
        if (!cold && b->a.handle == signal_handle) continue;
        b->data = aligned_alloc(4096, (b->a.size + 4095) & ~4095ull);
        int ok = 0;
        if (use_hsa && b->a.va_addr) {
            void *agent_ptr = NULL;
            if (!h_lock(b->data, b->a.size, &gpu_agent, 1, &agent_ptr)) {
                h_sstore(sig, 1);
                if (!h_acopy(agent_ptr, cpu_agent, (void *)(uintptr_t)b->a.va_addr, gpu_agent, b->a.size, 0, NULL, sig))
                    ok = h_swait(sig, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) == 0;
                h_unlock(b->data);
            }
        }
        if (!ok) {
            void *m = real_mmap(NULL, b->a.size, PROT_READ, MAP_SHARED, render_fd, b->a.mmap_offset);
            if (m != MAP_FAILED) {
                memcpy(b->data, m, b->a.size);
                real_munmap(m, b->a.size);
                ok = 1;
            }
        }
        if (ok) {
            *bytes += b->a.size;
        } else {
            (*nlost)++;
            free(b->data);
            b->data = NULL;
        }
    }
    if (use_hsa) h_sdestroy(sig);
}

/* ---- mapping audit against /proc/self/maps ----------------------------------- */
/*
 * The kernel's view is the source of truth at checkpoint time. A record stays
 * live only if its whole range is a mapping of the same device file at the
 * expected offset. A kernel device mapping not covered by the model is a
 * refusal (fail-closed).
 */
struct pm { uintptr_t s, e; uint64_t off; char path[64]; };

static int audit_maps(void) {
    static struct pm v[65536];
    int nv = 0;
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return -1;
    char line[512];
    while (fgets(line, sizeof line, f) && nv < 65536) {
        unsigned long s0, e0, off, ino;
        char perm[8], dev[16], path[256] = "";
        if (sscanf(line, "%lx-%lx %7s %lx %15s %lu %255s", &s0, &e0, perm, &off, dev, &ino, path) < 6) continue;
        if (strncmp(path, "/dev/kfd", 8) && strncmp(path, "/dev/dri/renderD", 16)) continue;
        v[nv].s = s0;
        v[nv].e = e0;
        v[nv].off = off;
        snprintf(v[nv].path, sizeof v[nv].path, "%s", path);
        nv++;
    }
    fclose(f);
    int dropped = 0, kept = 0;
    for (int i = 0; i < nmp; i++) {
        struct mp *p = &mps[i];
        if (!p->live) continue;
        uintptr_t a = (uintptr_t)p->addr, end = a + p->len;
        int ok = 1;
        while (a < end && ok) {
            ok = 0;
            for (int k = 0; k < nv; k++)
                if (v[k].s <= a && a < v[k].e && !strcmp(v[k].path, fdpath[p->fd]) &&
                    v[k].off + (a - v[k].s) == p->off + (a - (uintptr_t)p->addr)) {
                    a = v[k].e;
                    ok = 1;
                    break;
                }
        }
        if (!ok) {
            p->live = 0;
            dropped++;
        } else {
            kept++;
        }
    }
    int uncovered = 0;
    for (int k = 0; k < nv; k++) {
        int cov = 0;
        for (int i = 0; i < nmp && !cov; i++) {
            struct mp *p = &mps[i];
            cov = p->live && (uintptr_t)p->addr <= v[k].s && v[k].e <= (uintptr_t)p->addr + p->len;
        }
        if (!cov) {
            uncovered++;
            if (uncovered <= 5)
                kfdr_log("  uncovered device mapping %lx-%lx %s off 0x%llx\n", (unsigned long)v[k].s, (unsigned long)v[k].e,
                         v[k].path, (unsigned long long)v[k].off);
        }
    }
    kfdr_log("  maps audit: %d kernel device mappings, model kept %d, dropped %d stale, %d uncovered\n", nv, kept, dropped,
             uncovered);
    return uncovered ? -1 : 0;
}

/* ---- pre-checks -------------------------------------------------------------- */

static int render_fd_used(void) {
    for (int j = nmp - 1; j >= 0; j--)
        if (mps[j].live && kfdr_kind_of(mps[j].fd) == FD_RENDER) return mps[j].fd;
    for (int fd = 0; fd < MAX_DEV_FD; fd++)
        if (kfdr_kind_of(fd) == FD_RENDER) return fd;
    return -1;
}

/* Group device fds that share one open file (libdrm may F_DUPFD the render
   node; GEM handles and mmap offsets belong to one DRM file). */
static void group_device_fds(void) {
    int ndev = 0, ndup = 0;
    pid_t me = getpid();
    for (int fd = 0; fd < MAX_DEV_FD; fd++) {
        fdlead[fd] = fd;
        if (fdkind[fd] != FD_KFD && fdkind[fd] != FD_RENDER) continue;
        ndev++;
        for (int l = 0; l < fd; l++)
            if ((fdkind[l] == FD_KFD || fdkind[l] == FD_RENDER) && fdlead[l] == l &&
                syscall(SYS_kcmp, me, me, KCMP_FILE, l, fd) == 0) {
                fdlead[fd] = l;
                ndup++;
                kfdr_dbg("  fd %d shares open file with fd %d (%s)\n", fd, l, fdpath[fd]);
                break;
            }
    }
    kfdr_log("  device fds: %d, of them %d dup(s) of another\n", ndev, ndup);
}

/* DRM fail-closed check. Returns 0 if the render-node state is replayable. */
static int check_drm(void) {
    unsigned long bad = 0;
    unsigned bad_nr = 0;
    for (int n = 0; n < 256; n++)
        if (drm_ok[n] || drm_err[n]) kfdr_dbg("  drm ioctl nr 0x%02x: ok %lu err %lu\n", n, drm_ok[n], drm_err[n]);
    for (int n = 0; n < 256; n++)
        if (drm_ok[n] && !kfdr_drm_nr_allowed(n)) {
            bad += drm_ok[n];
            bad_nr = n;
        }
    int ndmabuf = 0;
    char lp[64], tgt[128];
    for (int fd = 0; fd < MAX_FD; fd++) {
        snprintf(lp, sizeof lp, "/proc/self/fd/%d", fd);
        ssize_t k = readlink(lp, tgt, sizeof tgt - 1);
        if (k > 0) {
            tgt[k] = 0;
            if (strstr(tgt, "dmabuf")) ndmabuf++;
        }
    }
    kfdr_log("  drm live state: %d GEM handle(s), %ld GEM_VA mapping(s), %d dmabuf fd(s), overflow %lu\n", ngem_live,
             gem_va_live, ndmabuf, drm_track_overflow);
    if (ngem_live || gem_va_live || ndmabuf || drm_track_overflow) {
        bad++;
        bad_nr = 0xff;
    }
    if (nsobj >= MAX_SYNCOBJ) {
        bad++;
        bad_nr = DRM_NR_SYNCOBJ_CREATE;
    }
    int live = 0;
    for (int i = 0; i < nsobj; i++) live += sobj[i].destroy ? -1 : 1;
    if (nsobj) kfdr_log("  drm syncobj: %d create/destroy events, %d live (replayed on cold restore)\n", nsobj, live);
    if (bad) {
        if (kfdr_env_flag("KFDR_DRM_ALLOW")) {
            kfdr_log("WARNING: %lu state-changing DRM ioctl(s), last nr 0x%02x (KFDR_DRM_ALLOW=1)\n", bad, bad_nr);
        } else {
            kfdr_fail("REFUSE: %lu state-changing DRM ioctl(s), last nr 0x%02x", bad, bad_nr);
            return -1;
        }
    }
    return 0;
}

/* ---- freeze ------------------------------------------------------------------ */

static void unfreeze(void) {
    pthread_mutex_lock(&mu);
    frozen = 0;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
}

static void signal_all_events(void) {
    for (int i = 0; i < nev; i++)
        if (evs[i].live) {
            struct ev_args e = {evs[i].a.event_id, 0};
            real_ioctl(kfd_fd, IOC_SET_EVENT, &e);
        }
}

/*
 * Stop every other thread from touching the driver: new KFD ioctls park at the
 * barrier; threads sleeping in WAIT_EVENTS are woken with SET_EVENT and park
 * on their way back in.
 *
 * ROCm 10.x: a woken ROCr event thread first polls signals in GTT from user
 * space and only then re-enters WAIT_EVENTS. Device mappings may only be
 * removed once every thread that has ever waited for events is calm: parked at
 * the barrier, or blocked in a syscall for 50 ms in a row
 * (/proc/self/task/<tid>/syscall is not "running"). A thread spinning in user
 * space for 5 s is a refusal.
 */
static int freeze_threads(void) {
    controller = pthread_self();
    frozen = 1;
    for (int round = 0; round < 100 && inflight_wait > 0; round++) {
        signal_all_events();
        usleep(10000);
    }
    if (inflight_wait) {
        kfdr_fail("REFUSE: %d thread(s) still in WAIT_EVENTS", inflight_wait);
        unfreeze();
        return -1;
    }
    double tp = kfdr_now(), calm_since = -1;
    int calm = 0, nalive = 0;
    while (kfdr_now() - tp < 5.0) {
        signal_all_events();
        usleep(2000);
        int busy = 0;
        nalive = 0;
        for (int i = 0; i < nwaiter; i++) {
            if (waiter_tid[i] == (long)syscall(SYS_gettid)) continue;
            char pth[64], st[32] = "";
            snprintf(pth, sizeof pth, "/proc/self/task/%ld/syscall", waiter_tid[i]);
            int f = open(pth, O_RDONLY);
            if (f < 0) continue;
            int r = read(f, st, sizeof st - 1);
            close(f);
            nalive++;
            if (r <= 0 || !strncmp(st, "running", 7)) busy++;
        }
        if (busy) {
            calm_since = -1;
            continue;
        }
        if (calm_since < 0) calm_since = kfdr_now();
        if (kfdr_now() - calm_since >= 0.05) {
            calm = 1;
            break;
        }
    }
    kfdr_log("  freeze: %d event-waiter thread(s), %d parked at barrier, settled=%d after %.3f s\n", nalive, parked, calm,
             kfdr_now() - tp);
    if (!calm) {
        kfdr_fail("REFUSE: event-waiter thread keeps running in user space");
        unfreeze();
        return -1;
    }
    return 0;
}

/* All queues must be empty (rptr == wptr). */
static int check_queues_idle(void) {
    for (int i = 0; i < nq; i++)
        if (qs[i].live) {
            uint64_t w = *(volatile uint64_t *)(uintptr_t)qs[i].a.wptr, r = *(volatile uint64_t *)(uintptr_t)qs[i].a.rptr;
            uint32_t slots = qs[i].a.ring_size / AQL_PACKET_SIZE;
            kfdr_dbg("  queue %u type %u L=%u wptr %llu (mod L = %llu) rptr %llu doorbell_off 0x%llx\n", qs[i].a.queue_id,
                     qs[i].a.queue_type, slots, (unsigned long long)w, (unsigned long long)(slots ? w % slots : 0),
                     (unsigned long long)r, (unsigned long long)qs[i].a.doorbell_offset);
            if (r != w) {
                kfdr_fail("REFUSE: queue %u busy (rptr %llu, wptr %llu)", qs[i].a.queue_id, (unsigned long long)r,
                          (unsigned long long)w);
                return -1;
            }
        }
    return 0;
}

/* ---- teardown ---------------------------------------------------------------- */

/* Warm: destroy queues, events and BOs in this process (fds stay open). */
static int teardown_warm(int *nqd, int *ned, int *nfreed) {
    for (int i = 0; i < nq; i++)
        if (qs[i].live) {
            struct dq_args d = {qs[i].a.queue_id, 0};
            if (kfdr_kio(IOC_DESTROY_QUEUE, &d, "DESTROY_QUEUE")) return -1;
            (*nqd)++;
        }
    for (int i = 0; i < nev; i++)
        if (evs[i].live) {
            struct ev_args e = {evs[i].a.event_id, 0};
            if (kfdr_kio(IOC_DESTROY_EVENT, &e, "DESTROY_EVENT")) return -1;
            (*ned)++;
        }
    for (int i = 0; i < nbo; i++) {
        struct bo *b = &bos[i];
        /* doorbell, MMIO and the signal page cannot be freed (EPERM) and stay in place */
        if (!b->live || (b->a.flags & (F_DOORBELL | F_MMIO)) || b->a.handle == signal_handle) continue;
        if (b->ngpu) {
            struct map_args u = {b->a.handle, (uint64_t)(uintptr_t)b->gpus, b->ngpu, 0};
            if (kfdr_kio(IOC_UNMAP, &u, "UNMAP")) return -1;
        }
        struct free_args f = {b->a.handle};
        if (kfdr_kio(IOC_FREE, &f, "FREE")) return -1;
        for (int j = 0; j < nmp; j++)
            if (kfdr_maps_bo(&mps[j], b))
                real_mmap(mps[j].addr, mps[j].len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
        (*nfreed)++;
    }
    return 0;
}

/*
 * Cold: kernel objects die with the process; hide everything DMTCP must not
 * see. Device mappings become anonymous PROT_NONE reservations, except small
 * GTT mappings (ROCr signal pools): those become an anonymous RW copy of their
 * current contents, because the ROCm 10.x event thread polls them from user
 * space between sleeps and may wake up between here and kfdr_restore()
 * (including right after dmtcp_restart). Restore maps the real BO back at the
 * same VA and writes the saved contents. Finally the device fds are closed.
 */
static void teardown_cold(void) {
    int nsnap = 0;
    size_t snap_bytes = 0;
    for (int j = 0; j < nmp; j++)
        if (mps[j].live) {
            struct bo *gb = NULL;
            for (int i = 0; i < nbo && !gb; i++)
                if (bos[i].live && (bos[i].a.flags & F_GTT) && bos[i].a.size <= (64u << 20) && kfdr_maps_bo(&mps[j], &bos[i]))
                    gb = &bos[i];
            if (gb && (mps[j].prot & PROT_READ)) {
                /* Atomic swap: fill anonymous memory at a temporary address, then
                   mremap it over the VA. A reader sees either the old BO or the
                   finished copy, never zeros. */
                void *tmp = real_mmap(NULL, mps[j].len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (tmp != MAP_FAILED) {
                    memcpy(tmp, mps[j].addr, mps[j].len);
                    if (mremap(tmp, mps[j].len, mps[j].len, MREMAP_MAYMOVE | MREMAP_FIXED, mps[j].addr) == mps[j].addr) {
                        mps[j].snap = 1;
                        nsnap++;
                        snap_bytes += mps[j].len;
                        continue;
                    }
                    kfdr_log("  snapshot mremap %p failed: %s\n", mps[j].addr, strerror(errno));
                    real_munmap(tmp, mps[j].len);
                }
            }
            real_mmap(mps[j].addr, mps[j].len, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED | MAP_NORESERVE, -1, 0);
        }
    kfdr_log("  cold: %d small GTT mappings (%.1f MiB) kept as anonymous RW snapshots, rest PROT_NONE\n", nsnap,
             snap_bytes / 1048576.0);
    for (int fd = 0; fd < MAX_DEV_FD; fd++) {
        cold_kind[fd] = (fdkind[fd] == FD_KFD || fdkind[fd] == FD_RENDER) ? fdkind[fd] : 0;
        if (cold_kind[fd]) {
            real_close(fd);
            fdkind[fd] = FD_UNCLASSIFIED;
        }
    }
    cold_state = 1;
}

/* ---- entry point ------------------------------------------------------------- */

KFDR_API int kfdr_checkpoint(int cold) {
    kfdr_resolve_real();
    kfdr_clear_error();
    double t0 = kfdr_now();
    for (int fd = 0; fd < MAX_DEV_FD; fd++) fdkind[fd] = FD_UNCLASSIFIED;
    for (int fd = 0; fd < MAX_DEV_FD; fd++) kfdr_kind_of(fd);
    render_fd = render_fd_used();
    if (!gpu_old) /* gpu_id of the first recorded BO that has one */
        for (int i = 0; i < nbo && !gpu_old; i++) gpu_old = bos[i].a.gpu_id;
    if (topo_old < 0) topo_old = kfdr_topo_node_of(gpu_old);
    kfdr_log("checkpoint (%s): kfd fd %d, render fd %d, gpu_id %u\n", cold ? "cold" : "warm", kfd_fd, render_fd, gpu_old);
    group_device_fds();
    if (partial_overlaps) kfdr_log("  WARNING: %lu partial mapping overlaps (records dropped whole)\n", partial_overlaps);
    if (check_drm()) return -1;
    if (unknown_calls) {
        kfdr_fail("REFUSE: %lu unknown KFD ioctl(s), last nr 0x%x", unknown_calls, unknown_nr);
        return -1;
    }

    /* 0. Contents, while the runtime is alive (ROCr copies with a blit kernel at
          near GPU speed). ROCr ioctls are not hidden during the dump: it may
          allocate new BOs (signal pool) and they must enter the model, or their
          VA would be empty in the new process. */
    double bytes = 0;
    int nlost = 0;
    double tc = kfdr_now();
    dump_contents(&nlost, &bytes, cold);
    double copy_s = kfdr_now() - tc;
    if (nlost) {
        kfdr_fail("REFUSE: %d BO(s) unreadable", nlost);
        return -1;
    }
    if (kfdr_debug_on()) {
        for (int i = 0; i < nbo; i++)
            if (bos[i].live)
                kfdr_log("  live BO h=0x%llx va=0x%llx size=0x%llx off=0x%llx flags=0x%08x\n", (unsigned long long)bos[i].a.handle,
                         (unsigned long long)bos[i].a.va_addr, (unsigned long long)bos[i].a.size,
                         (unsigned long long)bos[i].a.mmap_offset, bos[i].a.flags);
        for (int j = 0; j < nmp; j++)
            if (mps[j].live)
                kfdr_log("  live map %p len 0x%zx fd %d off 0x%llx\n", mps[j].addr, mps[j].len, mps[j].fd,
                         (unsigned long long)mps[j].off);
    }
    if (audit_maps()) {
        kfdr_fail("REFUSE: device mappings not covered by the model");
        return -1;
    }

    /* 1. Freeze. */
    if (freeze_threads()) return -1;
    /* 2. Queues must be empty. */
    if (check_queues_idle()) {
        unfreeze();
        return -1;
    }
    int nqd = 0, ned = 0, nfreed = 0;
    if (!cold) {
        if (teardown_warm(&nqd, &ned, &nfreed)) return -1; /* stays frozen: objects are half destroyed */
    } else {
        teardown_cold();
    }
    kfdr_log("checkpoint done: %.1f MiB copied in %.3f s (%s); %d queues, %d events, %d BO freed; total %.3f s\n",
             bytes / 1048576, copy_s, copy_mode, nqd, ned, nfreed, kfdr_now() - t0);
    return 0;
}
