/*
 * kfdr_restore(): reopen devices and replay recorded objects at the same VAs and handles.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kfdr.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

/* ---- device fds ------------------------------------------------------------ */

/* Open each open file description once ("leaders"), on the original fd number. */
static int reopen_leaders(void) {
    for (int fd = 0; fd < MAX_DEV_FD; fd++)
        if (cold_kind[fd] && fdlead[fd] == fd) {
            int t = open(fdpath[fd], O_RDWR | O_CLOEXEC);
            if (t < 0 && errno == ENOENT && !strncmp(fdpath[fd], "/dev/dri/renderD", 16)) {
                /* Another GPU: the container has one render node, under a different number. */
                for (int m = 128; m < 256 && t < 0; m++) {
                    char p[64];
                    snprintf(p, sizeof p, "/dev/dri/renderD%d", m);
                    t = open(p, O_RDWR | O_CLOEXEC);
                    if (t >= 0) {
                        kfdr_log("cold: render node %s -> %s\n", fdpath[fd], p);
                        snprintf(fdpath[fd], sizeof fdpath[fd], "%s", p);
                    }
                }
            }
            if (t < 0) {
                kfdr_fail("open %s: %s", fdpath[fd], strerror(errno));
                return -1;
            }
            if (t != fd) {
                if (dup3(t, fd, O_CLOEXEC) < 0) {
                    kfdr_fail("dup3 %d -> %d: %s", t, fd, strerror(errno));
                    return -1;
                }
                real_close(t);
            }
        }
    return 0;
}

/* Then dup3 the other fds of each group onto their leader. */
static int reopen_followers(void) {
    for (int fd = 0; fd < MAX_DEV_FD; fd++)
        if (cold_kind[fd] && fdlead[fd] != fd) {
            if (dup3(fdlead[fd], fd, O_CLOEXEC) < 0) {
                kfdr_fail("dup3 follower %d -> %d: %s", fdlead[fd], fd, strerror(errno));
                return -1;
            }
        }
    for (int fd = 0; fd < MAX_DEV_FD; fd++)
        if (cold_kind[fd]) fdkind[fd] = cold_kind[fd];
    return 0;
}

/* ---- process-level state ------------------------------------------------------ */

/* Live syncobjs are re-created in the original order and must get the same
   handle; destroyed ones are created and destroyed too, so that the kernel idr
   hands out the same numbers. */
static int replay_syncobjs(void) {
    for (int i = 0; i < nsobj; i++) {
        if (sobj[i].destroy) {
            uint32_t a[2] = {sobj[i].handle, 0};
            if (real_ioctl(sobj[i].fd, _IOWR('d', DRM_NR_SYNCOBJ_DESTROY, uint32_t[2]), a)) {
                kfdr_fail("syncobj destroy: %s", strerror(errno));
                return -1;
            }
            continue;
        }
        uint32_t a[2] = {0, sobj[i].flags};
        if (real_ioctl(sobj[i].fd, _IOWR('d', DRM_NR_SYNCOBJ_CREATE, uint32_t[2]), a)) {
            kfdr_fail("syncobj re-create: %s", strerror(errno));
            return -1;
        }
        if (a[0] != sobj[i].handle) {
            kfdr_fail("syncobj handle mismatch: got %u want %u", a[0], sobj[i].handle);
            return -1;
        }
    }
    if (nsobj) kfdr_log("cold: %d drm syncobj re-created, handles match\n", nsobj);
    return 0;
}

/* ACQUIRE_VM first (nothing can be allocated or mapped without a VM), then the
   other process-level settings in their original order. SET_TRAP_HANDLER and
   SET_SCRATCH_BACKING_VA reference VAs and are replayed after the BOs. */
static int replay_process_settings(void) {
    if (plv[NR_ACQUIRE_VM].set && kfdr_kio(plv[NR_ACQUIRE_VM].req, plv[NR_ACQUIRE_VM].arg, "ACQUIRE_VM")) return -1;
    for (int i = 0; i < npl; i++) {
        int nr = pl_order[i];
        if (nr == NR_ACQUIRE_VM || nr == NR_SET_TRAP_HANDLER || nr == NR_SET_SCRATCH_BACKING_VA) continue;
        char w[32];
        snprintf(w, sizeof w, "process-level 0x%x", nr);
        if (kfdr_kio(plv[nr].req, plv[nr].arg, w)) return -1;
    }
    kfdr_log("cold: ACQUIRE_VM + %d process-level ioctls replayed\n", npl);
    return 0;
}

/* ---- buffer objects ------------------------------------------------------------ */

static int by_handle(const void *x, const void *y) {
    const struct bo *a = *(struct bo *const *)x, *b = *(struct bo *const *)y;
    return (a->a.handle > b->a.handle) - (a->a.handle < b->a.handle);
}

static struct bo *order[MAX_BO];

/*
 * Re-create BOs in ascending handle order. The kernel hands out handles in
 * sequence, so holes left by freed BOs are filled with 4 KiB dummies that are
 * freed afterwards. Each BO gets the same va_addr, flags and userptr address,
 * its GPU mappings, its contents, and its CPU mappings at the old addresses
 * with the new mmap offset.
 */
static int recreate_bos(int cold, int *nbo_out, int *ndummy_out, double *wbytes) {
    int n = 0;
    for (int i = 0; i < nbo; i++) {
        struct bo *b = &bos[i];
        if (!b->live) continue;
        if (!cold && ((b->a.flags & (F_DOORBELL | F_MMIO)) || b->a.handle == signal_handle)) continue;
        order[n++] = b;
    }
    qsort(order, n, sizeof *order, by_handle);
    /* Resolve the owner of every mapping by the OLD offsets, before anything is
       re-created: new fake offsets of some BOs may fall into old ranges of
       others (after free/realloc in a caching allocator). */
    for (int j = 0; j < nmp; j++) {
        mps[j].owner = NULL;
        for (int i = 0; i < n && !mps[j].owner; i++)
            if (kfdr_maps_bo(&mps[j], order[i])) mps[j].owner = order[i];
    }
    static uint64_t dummies[MAX_BO];
    int nd = 0;
    int nocopy = kfdr_env_flag("KFDR_NOCOPY");
    for (int i = 0; i < n; i++) {
        struct bo *b = order[i];
        for (;;) {
            struct alloc_args d = {0, 4096, 0, 0, b->a.gpu_id, F_VRAM | F_WRITABLE};
            if (kfdr_kio(IOC_ALLOC, &d, "ALLOC dummy")) return -1;
            if ((uint32_t)d.handle < (uint32_t)b->a.handle) {
                dummies[nd++] = d.handle;
                continue;
            }
            struct free_args f = {d.handle};
            kfdr_kio(IOC_FREE, &f, "FREE dummy");
            if ((uint32_t)d.handle > (uint32_t)b->a.handle) {
                kfdr_fail("handle 0x%llx taken", (unsigned long long)b->a.handle);
                return -1;
            }
            break;
        }
        struct alloc_args a = b->a;
        a.handle = 0;
        a.mmap_offset = (b->a.flags & F_USERPTR) ? b->user_addr : 0;
        if (kfdr_kio(IOC_ALLOC, &a, "ALLOC")) {
            kfdr_log("  flags 0x%x va 0x%llx\n", b->a.flags, (unsigned long long)b->a.va_addr);
            return -1;
        }
        if (a.handle != kfdr_th(b->a.handle)) {
            kfdr_fail("handle mismatch 0x%llx vs 0x%llx", (unsigned long long)a.handle, (unsigned long long)b->a.handle);
            return -1;
        }
        if (b->a.handle == signal_handle) signal_handle = a.handle;
        b->a.handle = a.handle;
        b->a.gpu_id = a.gpu_id;
        b->new_off = a.mmap_offset;
        if (b->ngpu) {
            struct map_args m = {a.handle, (uint64_t)(uintptr_t)b->gpus, b->ngpu, 0};
            if (kfdr_kio(IOC_MAP, &m, "MAP")) return -1;
        }
        if (b->a.flags & F_USERPTR) continue;
        /* Contents are written BEFORE the BO is mapped over its VA: ROCr threads
           may read signals during restore, and MAP_FIXED must replace the
           snapshot with an already filled BO. For GTT snapshots the source is
           the snapshot itself (it is newer than the dump). */
        if (b->data)
            for (int j = 0; j < nmp; j++)
                if (mps[j].live && mps[j].owner == b && mps[j].snap) {
                    uint64_t o = mps[j].off - b->a.mmap_offset;
                    if (o + mps[j].len <= b->a.size) memcpy((char *)b->data + o, mps[j].addr, mps[j].len);
                }
        if (b->data && !nocopy) {
            void *m = real_mmap(NULL, b->a.size, PROT_WRITE, MAP_SHARED, render_fd, b->new_off);
            if (m == MAP_FAILED) {
                kfdr_fail("cannot write BO: %s", strerror(errno));
                return -1;
            }
            memcpy(m, b->data, b->a.size);
            real_munmap(m, b->a.size);
            *wbytes += b->a.size;
        }
        for (int j = 0; j < nmp; j++)
            if (mps[j].live && mps[j].owner == b) {
                struct mp *p = &mps[j];
                void *r = real_mmap(p->addr, p->len, p->prot, p->flags | MAP_FIXED, p->fd, b->new_off + (p->off - b->a.mmap_offset));
                if (r != p->addr) {
                    kfdr_fail("remap %p failed: %s (len 0x%zx prot %d flags 0x%x fd %d kind %d off 0x%llx; BO h=0x%llx va 0x%llx "
                              "size 0x%llx flags 0x%x old_off 0x%llx new_off 0x%llx)",
                              p->addr, strerror(errno), p->len, p->prot, p->flags, p->fd, fdkind[p->fd], (unsigned long long)p->off,
                              (unsigned long long)b->a.handle, (unsigned long long)b->a.va_addr, (unsigned long long)b->a.size,
                              b->a.flags, (unsigned long long)b->a.mmap_offset, (unsigned long long)b->new_off);
                    return -1;
                }
                p->off = b->new_off + (p->off - b->a.mmap_offset);
                p->snap = 0;
            }
        if (b->data) {
            free(b->data);
            b->data = NULL;
        }
        b->a.mmap_offset = b->new_off;
    }
    for (int i = 0; i < nd; i++) {
        struct free_args f = {dummies[i]};
        kfdr_kio(IOC_FREE, &f, "FREE dummy");
    }
    if (cold) {
        kfdr_log("cold: %d BO recreated\n", n);
        if (kfdr_debug_on())
            for (int i = 0; i < n; i++)
                kfdr_log("  BO h=0x%llx va=0x%llx size=0x%llx flags=0x%08x ngpu=%u%s\n", (unsigned long long)order[i]->a.handle,
                         (unsigned long long)order[i]->a.va_addr, (unsigned long long)order[i]->a.size, order[i]->a.flags,
                         order[i]->ngpu, order[i]->a.handle == signal_handle ? " [signal page]" : "");
        for (int j = 0; j < nmp; j++) {
            struct mp *p = &mps[j];
            if (p->live && !p->owner)
                kfdr_dbg("cold: mapping %p len %zu fd-kind %d off 0x%llx has no BO (doorbell, remapped with its queue)\n", p->addr,
                         p->len, fdkind[p->fd], (unsigned long long)p->off);
        }
    }
    *nbo_out = n;
    *ndummy_out = nd;
    return 0;
}

/* ---- events -------------------------------------------------------------------- */

/* Re-create events in ascending id order (holes filled with dummy events);
   on a cold restore the first one also re-creates the signal page. */
static int recreate_events(int cold, int *ne_out) {
    int ne = 0, page_done = !cold;
    static char edone[MAX_EVENT];
    memset(edone, 0, sizeof edone);
    for (;;) {
        int k = -1;
        for (int i = 0; i < nev; i++)
            if (evs[i].live && !edone[i] && (k < 0 || evs[i].a.event_id < evs[k].a.event_id)) k = i;
        if (k < 0) break;
        edone[k] = 1;
        static uint32_t edum[MAX_EVENT];
        int ndum = 0;
        for (;;) {
            struct ce_args c = evs[k].a;
            c.event_id = 0;
            c.slot = 0;
            c.trigger = 0;
            c.page_offset = page_done ? 0 : signal_handle;
            if (kfdr_kio(IOC_CREATE_EVENT, &c, "CREATE_EVENT")) return -1;
            page_done = 1;
            if (c.event_id < evs[k].a.event_id) {
                edum[ndum++] = c.event_id;
                continue;
            }
            if (c.event_id != evs[k].a.event_id) {
                kfdr_fail("event id %u != %u", c.event_id, evs[k].a.event_id);
                return -1;
            }
            break;
        }
        for (int i = 0; i < ndum; i++) {
            struct ev_args e = {edum[i], 0};
            kfdr_kio(IOC_DESTROY_EVENT, &e, "DESTROY dummy event");
        }
        ne++;
    }
    *ne_out = ne;
    return 0;
}

/* ---- queues -------------------------------------------------------------------- */

/*
 * The new hardware queue starts at rptr = 0 while ROCr is already at N. Drive
 * the command processor through N no-op packets, in laps of L = ring slots: in
 * each lap only slots with indices [cur, end) are valid (BARRIER_AND), all
 * others are INVALID, so the CP cannot run past N (on hardware it follows
 * packet headers, not the doorbell wptr). ~1M packets/s on MI350X.
 */
static int fast_forward(const struct cq_args *c, uint32_t old_id, uint64_t N, double *ms) {
    uint8_t *aq = (uint8_t *)(uintptr_t)(c->wptr - 56); /* amd_queue_t: write_dispatch_id at +56 */
    if (*(uint64_t *)(aq + 8) != c->ring) {
        kfdr_fail("amd_queue_t layout mismatch");
        return -1;
    }
    volatile uint64_t *doorbell = *(volatile uint64_t **)((uint8_t *)(uintptr_t)*(uint64_t *)(aq + 16) + 8);
    volatile uint64_t *rp = (volatile uint64_t *)(uintptr_t)c->rptr;
    uint32_t slots = c->ring_size / AQL_PACKET_SIZE;
    uint8_t *ring = (uint8_t *)(uintptr_t)c->ring;
    *rp = 0;
    double t = kfdr_now();
    uint64_t cur = 0;
    int laps = 0;
    for (uint32_t s = 0; s < slots; s++) memset(ring + s * AQL_PACKET_SIZE + 2, 0, AQL_PACKET_SIZE - 2);
    while (cur < N) {
        uint64_t end = cur + slots < N ? cur + slots : N;
        for (uint32_t s = 0; s < slots; s++)
            __atomic_store_n((uint16_t *)(ring + s * AQL_PACKET_SIZE), (uint16_t)AQL_HDR_INVALID, __ATOMIC_RELAXED);
        for (uint64_t i = cur; i < end; i++)
            __atomic_store_n((uint16_t *)(ring + (i % slots) * AQL_PACKET_SIZE), (uint16_t)AQL_HDR_BARRIER_AND, __ATOMIC_RELAXED);
        __sync_synchronize();
        *doorbell = end - 1;
        double ts = kfdr_now();
        while (*rp < end && kfdr_now() - ts < 5) usleep(20);
        if (*rp != end) {
            kfdr_fail("queue %u fast-forward stuck at %llu/%llu (lap %d, end %llu)", old_id, (unsigned long long)*rp,
                      (unsigned long long)N, laps, (unsigned long long)end);
            return -1;
        }
        cur = end;
        laps++;
    }
    for (uint32_t s = 0; s < slots; s++)
        __atomic_store_n((uint16_t *)(ring + s * AQL_PACKET_SIZE), (uint16_t)AQL_HDR_INVALID, __ATOMIC_RELAXED);
    __sync_synchronize();
    double dt = (kfdr_now() - t) * 1e3;
    kfdr_dbg("  queue %u: fast-forward %llu packets, %d laps, %.2f ms\n", old_id, (unsigned long long)N, laps, dt);
    *ms += dt;
    return 0;
}

/* Re-create queues in ascending id order; remap the doorbell page; fast-forward AQL rings. */
static int recreate_queues(int cold, int *nq_out, double *ff_ms) {
    int nqc = 0;
    static char qdone[MAX_QUEUE];
    memset(qdone, 0, sizeof qdone);
    for (;;) {
        int k = -1;
        for (int i = 0; i < nq; i++)
            if (qs[i].live && !qdone[i] && (k < 0 || qs[i].a.queue_id < qs[k].a.queue_id)) k = i;
        if (k < 0) break;
        qdone[k] = 1;
        struct cq_args c = qs[k].a;
        uint32_t old_id = c.queue_id;
        uint64_t old_db = c.doorbell_offset;
        c.queue_id = 0;
        c.doorbell_offset = 0;
        uint64_t N = *(volatile uint64_t *)(uintptr_t)c.wptr;
        if (kfdr_kio(IOC_CREATE_QUEUE, &c, "CREATE_QUEUE")) return -1;
        if (c.queue_id != old_id) {
            kfdr_fail("queue id %u != %u", c.queue_id, old_id);
            return -1;
        }
        if ((c.doorbell_offset & 0xffffffffull) != (old_db & 0xffffffffull))
            kfdr_dbg("  queue %u doorbell offset changed 0x%llx -> 0x%llx\n", old_id, (unsigned long long)old_db,
                     (unsigned long long)c.doorbell_offset);
        qs[k].a.doorbell_offset = c.doorbell_offset;
        if (cold) {
            /* The doorbell page is mapped through /dev/kfd at the queue's doorbell_offset, not via a BO. */
            uint64_t oldp = old_db & ~0x1fffull, newp = c.doorbell_offset & ~0x1fffull;
            for (int j = 0; j < nmp; j++) {
                struct mp *p = &mps[j];
                if (!p->live || fdkind[p->fd] != FD_KFD || p->off != oldp) continue;
                void *r = real_mmap(p->addr, p->len, p->prot, p->flags | MAP_FIXED, p->fd, newp);
                if (r != p->addr) {
                    kfdr_fail("doorbell remap failed: %s", strerror(errno));
                    return -1;
                }
                kfdr_log("cold: doorbell page %p remapped (offset 0x%llx -> 0x%llx)\n", p->addr, (unsigned long long)oldp,
                         (unsigned long long)newp);
                p->off = newp;
            }
        }
        if (c.queue_type == QUEUE_TYPE_COMPUTE_AQL && N && fast_forward(&c, old_id, N, ff_ms)) return -1;
        nqc++;
    }
    *nq_out = nqc;
    return 0;
}

/* ---- entry point ------------------------------------------------------------- */

static void release_threads(void) {
    pthread_mutex_lock(&mu);
    frozen = 0;
    pthread_cond_broadcast(&cv);
    pthread_mutex_unlock(&mu);
}

KFDR_API int kfdr_restore(void) {
    kfdr_resolve_real();
    kfdr_clear_error();
    double t0 = kfdr_now();
    int cold = cold_state;
    if (cold) {
        uint32_t now_id = kfdr_sysfs_gpu_id();
        if (now_id && gpu_old && now_id != gpu_old) {
            gpu_new = now_id;
            kfdr_log("cold: different GPU: gpu_id %u -> %u, translating ioctl args and BO handles\n", gpu_old, gpu_new);
            topo_new = kfdr_topo_node_of(gpu_new);
            kfdr_log("cold: topology node %d -> %d (sysfs paths translated)\n", topo_old, topo_new);
        }
        if (reopen_leaders() || reopen_followers()) return -1;
        kfdr_log("cold: fds reopened (kfd %d, render %d)\n", kfd_fd, render_fd);
        if (replay_syncobjs()) return -1;
        if (replay_process_settings()) return -1;
    }
    int n = 0, nd = 0, ne = 0, nqc = 0;
    double wbytes = 0, ff_ms = 0;
    if (recreate_bos(cold, &n, &nd, &wbytes)) return -1;
    if (cold) { /* trap handler and scratch reference VAs: after the BOs */
        if (plv[NR_SET_TRAP_HANDLER].set &&
            kfdr_kio(plv[NR_SET_TRAP_HANDLER].req, plv[NR_SET_TRAP_HANDLER].arg, "SET_TRAP_HANDLER"))
            return -1;
        if (plv[NR_SET_SCRATCH_BACKING_VA].set &&
            kfdr_kio(plv[NR_SET_SCRATCH_BACKING_VA].req, plv[NR_SET_SCRATCH_BACKING_VA].arg, "SET_SCRATCH_BACKING_VA"))
            return -1;
    }
    if (recreate_events(cold, &ne)) return -1;
    if (recreate_queues(cold, &nqc, &ff_ms)) return -1;
    kfdr_log("restore (%s): %d BO (%d dummies), %.1f MiB written%s, %d events, %d queues (fast-forward %.2f ms) in %.3f s\n",
             cold ? "cold" : "warm", n, nd, wbytes / 1048576, kfdr_env_flag("KFDR_NOCOPY") ? " [NOCOPY]" : "", ne, nqc, ff_ms,
             kfdr_now() - t0);
    if (cold) {
        restored_cold = 1;
        kfdr_trace_refresh();
    }
    /* Diagnostics: KFDR_PAUSE_FILE on the restore host -> pause 40 s to attach a debugger. */
    if (cold && access(KFDR_PAUSE_FILE, F_OK) == 0) {
        kfdr_log("pause for debugger: real pid %ld\n", (long)syscall(SYS_getpid));
        sleep(40);
    }
    cold_state = 0;
    kfdr_clear_error(); /* non-fatal failures above (dummy frees) do not count */
    release_threads();
    return 0;
}
