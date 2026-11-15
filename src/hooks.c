/*
 * libc interposition: record KFD/DRM objects and device mappings.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Intercepted: ioctl (on /dev/kfd and render nodes), mmap/mmap64/munmap,
 * close, open/open64/openat/openat64, fopen/fopen64, access.
 */
#include "kfdr.h"

#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

/* ---- fd classification ----------------------------------------------------- */

/* FD_KFD, FD_RENDER, or 0 for anything else. Cached in fdkind[] until close(). */
int kfdr_kind_of(int fd) {
    if (fd < 0 || fd >= MAX_FD) return 0;
    if (!fdkind[fd]) {
        char l[64], p[256];
        snprintf(l, sizeof l, "/proc/self/fd/%d", fd);
        ssize_t n = readlink(l, p, sizeof p - 1);
        fdkind[fd] = FD_OTHER;
        if (n > 0) {
            p[n] = 0;
            if (!strcmp(p, "/dev/kfd")) {
                fdkind[fd] = FD_KFD;
                kfd_fd = fd;
            } else if (!strncmp(p, "/dev/dri/renderD", 16)) {
                fdkind[fd] = FD_RENDER;
            }
            if (fdkind[fd] != FD_OTHER && fd < MAX_DEV_FD) snprintf(fdpath[fd], sizeof fdpath[fd], "%.63s", p);
        }
    }
    return fdkind[fd] == FD_OTHER ? 0 : fdkind[fd];
}

KFDR_API int close(int fd) {
    kfdr_resolve_real();
    if (fd >= 0 && fd < MAX_FD) fdkind[fd] = FD_UNCLASSIFIED;
    return real_close(fd);
}

/* ---- open/fopen/access: topology path rewriting, post-restore diagnostics --- */

/* After a cold restore, failed opens under /sys and /dev are always logged
   (diagnoses restore on another GPU: the old GPU's topology node is not
   readable in the new container). */
static void log_open_fail(const char *p, int e) {
    if (restored_cold && p && (!strncmp(p, "/sys/", 5) || !strncmp(p, "/dev/", 5))) {
        static int n;
        if (n++ < 40) kfdr_log("post-restore open %s failed: %s\n", p, strerror(e));
    }
}

static int (*real_openat)(int, const char *, int, ...);

static int do_openat(int dfd, const char *p, int fl, mode_t m) {
    if (!real_openat) real_openat = dlsym(RTLD_NEXT, "openat");
    char tb[512];
    p = kfdr_topo_xlate(p, tb, sizeof tb);
    int r = real_openat(dfd, p, fl, m);
    if (r < 0) {
        int e = errno;
        log_open_fail(p, e);
        errno = e;
    }
    return r;
}

#define MODE_ARG(fl, m)                                \
    do {                                               \
        if ((fl) & (O_CREAT | O_TMPFILE)) {            \
            va_list ap;                                \
            va_start(ap, fl);                          \
            m = va_arg(ap, int);                       \
            va_end(ap);                                \
        }                                              \
    } while (0)

KFDR_API int open(const char *p, int fl, ...) { mode_t m = 0; MODE_ARG(fl, m); return do_openat(AT_FDCWD, p, fl, m); }
KFDR_API int open64(const char *p, int fl, ...) { mode_t m = 0; MODE_ARG(fl, m); return do_openat(AT_FDCWD, p, fl, m); }
KFDR_API int openat(int d, const char *p, int fl, ...) { mode_t m = 0; MODE_ARG(fl, m); return do_openat(d, p, fl, m); }
KFDR_API int openat64(int d, const char *p, int fl, ...) { mode_t m = 0; MODE_ARG(fl, m); return do_openat(d, p, fl, m); }

/* glibc fopen does not go through open(), so it is hooked separately. */
static FILE *do_fopen(FILE *(*rf)(const char *, const char *), const char *p, const char *mode) {
    char tb[512];
    p = kfdr_topo_xlate(p, tb, sizeof tb);
    FILE *f = rf(p, mode);
    if (!f) {
        int e = errno;
        log_open_fail(p, e);
        errno = e;
    }
    return f;
}

KFDR_API FILE *fopen(const char *p, const char *mode) {
    static FILE *(*rf)(const char *, const char *);
    if (!rf) rf = dlsym(RTLD_NEXT, "fopen");
    return do_fopen(rf, p, mode);
}

KFDR_API FILE *fopen64(const char *p, const char *mode) {
    static FILE *(*rf)(const char *, const char *);
    if (!rf) rf = dlsym(RTLD_NEXT, "fopen64");
    return do_fopen(rf, p, mode);
}

KFDR_API int access(const char *p, int mode) {
    static int (*ra)(const char *, int);
    if (!ra) ra = dlsym(RTLD_NEXT, "access");
    char tb[512];
    p = kfdr_topo_xlate(p, tb, sizeof tb);
    return ra(p, mode);
}

/* ---- mmap/munmap: model of device mappings --------------------------------- */

/* Mark every live record overlapping [a, a+len) dead. */
static void kill_overlapping(void *a, size_t len) {
    for (int i = 0; i < nmp; i++)
        if (mps[i].live && (char *)mps[i].addr < (char *)a + len && (char *)a < (char *)mps[i].addr + mps[i].len) {
            if ((char *)mps[i].addr < (char *)a || (char *)mps[i].addr + mps[i].len > (char *)a + len) partial_overlaps++;
            mps[i].live = 0;
        }
}

KFDR_API void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    kfdr_resolve_real();
    void *r = real_mmap(addr, len, prot, flags, fd, off);
    if (r == MAP_FAILED && restored_cold) {
        int e = errno;
        static int n;
        if (n++ < 40)
            kfdr_log("post-restore mmap(%p, 0x%zx, prot %d, flags 0x%x, fd %d, off 0x%llx) failed: %s\n", addr, len, prot,
                     flags, fd, (unsigned long long)off, strerror(e));
        errno = e;
    }
    if (kfdr_trace_on() && !(flags & MAP_ANONYMOUS) && fd >= 0 && kfdr_kind_of(fd)) {
        int e = errno;
        kfdr_log("trace mmap(%p, 0x%zx, prot %d, flags 0x%x, fd %d, off 0x%llx) -> %p%s%s\n", addr, len, prot, flags, fd,
                 (unsigned long long)off, r, r == MAP_FAILED ? " errno " : "", r == MAP_FAILED ? strerror(e) : "");
        errno = e;
    }
    if (r != MAP_FAILED && !frozen) {
        /*
         * Any mmap over a recorded mapping (including the anonymous MAP_FIXED
         * ROCr uses to "unmap" a BO) kills the old record: otherwise the fake
         * offset of a freed BO, reused by DRM, would bind a stale mapping to a
         * new BO.
         */
        int dev = !(flags & MAP_ANONYMOUS) && kfdr_kind_of(fd);
        pthread_mutex_lock(&mu);
        kill_overlapping(r, len);
        if (dev && nmp < MAX_MAP) mps[nmp++] = (struct mp){r, len, prot, flags, fd, (uint64_t)off, 1, NULL, 0};
        pthread_mutex_unlock(&mu);
    }
    return r;
}
KFDR_API void *mmap64(void *, size_t, int, int, int, off_t) __attribute__((alias("mmap")));

KFDR_API int munmap(void *addr, size_t len) {
    kfdr_resolve_real();
    if (!frozen) {
        pthread_mutex_lock(&mu);
        kill_overlapping(addr, len);
        pthread_mutex_unlock(&mu);
    }
    return real_munmap(addr, len);
}

/* Does live mapping p (of /dev/kfd or a render node) fall inside BO b's mmap offset range? */
int kfdr_maps_bo(const struct mp *p, const struct bo *b) {
    if (!p->live) return 0;
    int k = p->fd >= 0 && p->fd < MAX_FD ? fdkind[p->fd] : 0;
    return (k == FD_KFD || k == FD_RENDER) && p->off >= b->a.mmap_offset && p->off < b->a.mmap_offset + b->a.size;
}

struct bo *kfdr_bo_by_handle(uint64_t h) {
    uint64_t h2 = kfdr_th(h); /* after restore on another GPU records hold handles with the new gpu_id */
    for (int i = nbo - 1; i >= 0; i--)
        if (bos[i].live && (bos[i].a.handle == h || bos[i].a.handle == h2)) return &bos[i];
    return NULL;
}

/* ---- ioctl ------------------------------------------------------------------ */

/* PRIME_FD2H of a buffer the fd already imported returns the same handle:
 * counted once. A handle beyond the bitmap cannot be tracked: refuse. */
static void gem_add(uint32_t h) {
    if (h >= MAX_GEM_HANDLE) {
        drm_track_overflow++;
        return;
    }
    uint64_t bit = 1ull << (h % 64);
    if (gem_live[h / 64] & bit) return;
    gem_live[h / 64] |= bit;
    ngem_live++;
}

static void gem_del(uint32_t h) {
    if (h >= MAX_GEM_HANDLE) return;
    uint64_t bit = 1ull << (h % 64);
    if (!(gem_live[h / 64] & bit)) return;
    gem_live[h / 64] &= ~bit;
    ngem_live--;
}

/* Block the calling thread until kfdr_restore() releases the freeze. */
static void park(void) {
    pthread_mutex_lock(&mu);
    parked++;
    while (frozen) pthread_cond_wait(&cv, &mu);
    parked--;
    pthread_mutex_unlock(&mu);
}

/* Render node: pass through, keep the DRM inventory and live-state counters. */
static int drm_ioctl(int fd, unsigned long req, void *arg) {
    int r = real_ioctl(fd, req, arg);
    int e = errno;
    unsigned n = _IOC_NR(req) & 0xff;
    const char *bt = getenv("KFDR_DRM_BT");
    if (bt && strtoul(bt, NULL, 0) == n) {
        void *fr[32];
        int k = backtrace(fr, 32);
        kfdr_log("drm ioctl 0x%02x on fd %d -> %d, backtrace:\n", n, fd, r);
        backtrace_symbols_fd(fr, k, 2);
    }
    if (r == 0 && n == DRM_NR_SYNCOBJ_CREATE && nsobj < MAX_SYNCOBJ) {
        uint32_t *a = arg; /* drm_syncobj_create { handle, flags } */
        pthread_mutex_lock(&mu);
        sobj[nsobj++] = (struct syncobj_rec){fd, a[0], a[1], 0};
        pthread_mutex_unlock(&mu);
    }
    if (r == 0 && (n == DRM_NR_PRIME_FD2H || n == DRM_NR_AMDGPU_GEM_CREATE || n == DRM_NR_GEM_CLOSE ||
                   n == DRM_NR_AMDGPU_GEM_VA)) {
        pthread_mutex_lock(&mu);
        if (n == DRM_NR_PRIME_FD2H || n == DRM_NR_AMDGPU_GEM_CREATE) gem_add(*(uint32_t *)arg); /* out handle at offset 0 */
        else if (n == DRM_NR_GEM_CLOSE) gem_del(*(uint32_t *)arg);
        else {
            uint32_t op = ((uint32_t *)arg)[2]; /* drm_amdgpu_gem_va.operation */
            if (op == AMDGPU_VA_OP_MAP) gem_va_live++;
            else if (op == AMDGPU_VA_OP_UNMAP) gem_va_live--;
            else drm_track_overflow++; /* REPLACE / CLEAR: not tracked -> refuse */
        }
        pthread_mutex_unlock(&mu);
    }
    if (r == 0 && n == DRM_NR_SYNCOBJ_DESTROY) {
        uint32_t h = *(uint32_t *)arg;
        pthread_mutex_lock(&mu);
        if (nsobj < MAX_SYNCOBJ) sobj[nsobj++] = (struct syncobj_rec){fd, h, 0, 1};
        pthread_mutex_unlock(&mu);
    }
    if (restored_cold && r) kfdr_log("post-restore DRM ioctl 0x%02x fd %d failed: %s\n", n, fd, strerror(e));
    if (kfdr_trace_on()) kfdr_log("trace drm ioctl 0x%02x fd %d -> %d%s%s\n", n, fd, r, r ? " errno " : "", r ? strerror(e) : "");
    if (r == 0) {
        __sync_fetch_and_add(&drm_ok[n], 1);
        if (restored_cold) __sync_fetch_and_add(&drm_ok_since_restore[n], 1);
    } else {
        __sync_fetch_and_add(&drm_err[n], 1);
    }
    errno = e;
    return r;
}

static void log_kfd_args(unsigned nr, void *arg) {
    if (nr == NR_ALLOC_MEMORY_OF_GPU) {
        struct alloc_args *a = arg;
        fprintf(stderr, " ALLOC va 0x%llx size 0x%llx flags 0x%08x gpu %u handle 0x%llx off 0x%llx", (unsigned long long)a->va_addr,
                (unsigned long long)a->size, a->flags, a->gpu_id, (unsigned long long)a->handle, (unsigned long long)a->mmap_offset);
    }
    if (nr == NR_MAP_MEMORY_TO_GPU || nr == NR_UNMAP_MEMORY_FROM_GPU) {
        struct map_args *m = arg;
        uint32_t *ids = (uint32_t *)(uintptr_t)m->dev_ptr;
        fprintf(stderr, " handle 0x%llx n %u dev0 %u succ %u", (unsigned long long)m->handle, m->n_devices,
                ids && m->n_devices ? ids[0] : 0, m->n_success);
    }
    if (nr == NR_FREE_MEMORY_OF_GPU) fprintf(stderr, " handle 0x%llx", (unsigned long long)((struct free_args *)arg)->handle);
}

/* Update the model after a successful KFD ioctl. Called with mu held. */
static void record_kfd(unsigned nr, unsigned long req, void *arg, const unsigned char *in_raw, size_t sz,
                       const struct alloc_args *in_alloc) {
    char c = kfdr_classify_kfd(nr);
    if (!c) {
        unknown_calls++;
        unknown_nr = nr;
    }
    if (c == 'P') {
        if (!plv[nr].set) pl_order[npl++] = nr;
        plv[nr].req = req;
        memcpy(plv[nr].arg, in_raw, sz);
        plv[nr].set = 1;
    }
    switch (nr) {
    case NR_ALLOC_MEMORY_OF_GPU:
        if (nbo < MAX_BO) {
            bos[nbo] = (struct bo){0};
            bos[nbo].a = *(struct alloc_args *)arg;
            if (in_alloc->flags & F_USERPTR) bos[nbo].user_addr = in_alloc->mmap_offset;
            bos[nbo].live = 1;
            nbo++;
        }
        break;
    case NR_FREE_MEMORY_OF_GPU: {
        struct bo *b = kfdr_bo_by_handle(((struct free_args *)arg)->handle);
        if (b) {
            for (int j = 0; j < nmp; j++)
                if (kfdr_maps_bo(&mps[j], b)) mps[j].live = 0;
            b->live = 0;
        }
        break;
    }
    case NR_MAP_MEMORY_TO_GPU: {
        struct map_args *m = arg;
        struct bo *b = kfdr_bo_by_handle(m->handle);
        if (b) {
            uint32_t *ids = (uint32_t *)(uintptr_t)m->dev_ptr;
            for (uint32_t i = 0; i < m->n_devices; i++) {
                int seen = 0;
                for (uint32_t j = 0; j < b->ngpu; j++) seen |= b->gpus[j] == ids[i];
                if (!seen && b->ngpu < 16) b->gpus[b->ngpu++] = ids[i];
            }
        }
        break;
    }
    case NR_UNMAP_MEMORY_FROM_GPU: {
        struct bo *b = kfdr_bo_by_handle(((struct map_args *)arg)->handle);
        if (b) b->ngpu = 0;
        break;
    }
    case NR_CREATE_QUEUE:
        if (nq < MAX_QUEUE) {
            qs[nq].a = *(struct cq_args *)arg;
            qs[nq].live = 1;
            nq++;
        }
        break;
    case NR_DESTROY_QUEUE:
        for (int i = 0; i < nq; i++)
            if (qs[i].live && qs[i].a.queue_id == ((struct dq_args *)arg)->queue_id) qs[i].live = 0;
        break;
    case NR_CREATE_EVENT: {
        struct ce_args *e = arg;
        uint64_t po = ((const struct ce_args *)in_raw)->page_offset; /* input: signal-page BO handle */
        int page = 0;
        if (!signal_handle && po && kfdr_bo_by_handle(po)) {
            signal_handle = po;
            page = 1;
        }
        if (nev < MAX_EVENT) {
            evs[nev].a = *e;
            evs[nev].a.page_offset = po;
            evs[nev].live = 1;
            evs[nev].page = page;
            nev++;
        }
        break;
    }
    case NR_DESTROY_EVENT:
        for (int i = 0; i < nev; i++)
            if (evs[i].live && evs[i].a.event_id == ((struct ev_args *)arg)->event_id) evs[i].live = 0;
        break;
    }
}

static int kfd_ioctl(int fd, unsigned long req, void *arg) {
    unsigned nr = _IOC_NR(req);
    if (frozen && !pthread_equal(pthread_self(), controller)) park();
    if (nr == NR_WAIT_EVENTS) {
        __sync_fetch_and_add(&inflight_wait, 1);
        long me = (long)syscall(SYS_gettid);
        int seen = 0;
        pthread_mutex_lock(&mu);
        for (int i = 0; i < nwaiter; i++) seen |= waiter_tid[i] == me;
        if (!seen && nwaiter < MAX_WAITER) waiter_tid[nwaiter++] = me;
        pthread_mutex_unlock(&mu);
    }
    int xl = gpu_new && restored_cold;
    struct xundo xu = {0};
    if (xl) kfdr_xlate(nr, arg, &xu);
    struct alloc_args in_alloc = {0};
    if (nr == NR_ALLOC_MEMORY_OF_GPU) in_alloc = *(struct alloc_args *)arg;
    unsigned char in_raw[128];
    size_t sz = _IOC_SIZE(req);
    if (sz > 128) sz = 128;
    memcpy(in_raw, arg, sz);

    int r = real_ioctl(fd, req, arg);
    int saved = errno;

    if (xl) {
        /* ALLOC: give the input gpu_id back, keep the output handle (the kernel knows it with the new gpu_id). */
        if (nr == NR_ALLOC_MEMORY_OF_GPU && xu.g) {
            *xu.g = xu.gv;
            xu.g = NULL;
        }
        kfdr_xundo_apply(&xu);
    }
    if (r == 0 && xl) kfdr_xlate_out(nr, arg);
    if (nr == NR_WAIT_EVENTS) __sync_fetch_and_sub(&inflight_wait, 1);

    if (kfdr_trace_on() && nr != NR_WAIT_EVENTS) {
        static int nt;
        if (nt++ < 20000) {
            fprintf(stderr, "[kfdr] trace KFD 0x%02x -> %d", nr, r);
            log_kfd_args(nr, arg);
            fprintf(stderr, "\n");
        }
    }
    /* After a cold restore every KFD failure is logged: the restored process
       runs with the snapshot's environment, so no flag could enable this. */
    if (r && restored_cold && nr != NR_WAIT_EVENTS) {
        static int nlog;
        if (nlog++ < 40) {
            fprintf(stderr, "[kfdr] post-restore KFD ioctl 0x%02x failed: %s (gpu_new %u)", nr, strerror(saved), gpu_new);
            log_kfd_args(nr, arg);
            fprintf(stderr, "\n");
        }
    }
    if (r == 0 && !frozen) {
        pthread_mutex_lock(&mu);
        record_kfd(nr, req, arg, in_raw, sz, &in_alloc);
        pthread_mutex_unlock(&mu);
    }
    errno = saved;
    return r;
}

KFDR_API int ioctl(int fd, unsigned long req, ...) {
    kfdr_resolve_real();
    va_list ap;
    va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);
    if (kfdr_kind_of(fd) == FD_RENDER) return drm_ioctl(fd, req, arg);
    /*
     * Between a cold checkpoint and reopen: an ioctl on the number of the
     * former /dev/kfd fd. If the number is closed, this is an application
     * thread: wait for the restore at the barrier (after reopen the fd is KFD
     * again). If another file now has that number (DMTCP's pagemap fd), let it
     * through.
     */
    if (cold_state && frozen && fd >= 0 && fd < MAX_DEV_FD && cold_kind[fd] == FD_KFD && kfdr_kind_of(fd) != FD_KFD) {
        char l[64], pth[64];
        snprintf(l, sizeof l, "/proc/self/fd/%d", fd);
        if (readlink(l, pth, sizeof pth - 1) < 0 && !pthread_equal(pthread_self(), controller)) {
            park();
            fdkind[fd] = FD_UNCLASSIFIED;
        }
    }
    if (kfdr_kind_of(fd) != FD_KFD) return real_ioctl(fd, req, arg);
    return kfd_ioctl(fd, req, arg);
}
