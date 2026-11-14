/*
 * Small public API functions and the KFDR_CRASH diagnostic handler.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kfdr.h"

#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

KFDR_API const char *kfdr_version(void) { return KFDR_VERSION_STRING; }

KFDR_API int kfdr_state(void) {
    if (cold_state) return KFDR_STATE_CHECKPOINTED_COLD;
    if (frozen) return KFDR_STATE_CHECKPOINTED_WARM;
    if (restored_cold) return KFDR_STATE_RESTORED;
    return KFDR_STATE_RUNNING;
}

KFDR_API uint64_t kfdr_max_wptr(uint32_t *slots) {
    uint64_t best = 0;
    uint32_t s = 0;
    for (int i = 0; i < nq; i++)
        if (qs[i].live && qs[i].a.queue_type == QUEUE_TYPE_COMPUTE_AQL) {
            uint64_t w = *(volatile uint64_t *)(uintptr_t)qs[i].a.wptr;
            if (w >= best) {
                best = w;
                s = qs[i].a.ring_size / AQL_PACKET_SIZE;
            }
        }
    if (slots) *slots = s;
    return best;
}

KFDR_API void kfdr_drm_report(void) {
    for (int n = 0; n < 256; n++)
        if (drm_ok_since_restore[n]) kfdr_log("  drm ioctl after restore nr 0x%02x: ok %lu\n", n, drm_ok_since_restore[n]);
    for (int n = 0; n < 256; n++)
        if (drm_err[n]) kfdr_log("  drm ioctl nr 0x%02x: errors total %lu\n", n, drm_err[n]);
}

/*
 * KFDR_CRASH=1: on SIGSEGV/SIGBUS/SIGABRT/SIGILL print the fault address, the
 * recorded device mapping it falls into (if any), the thread name and a
 * backtrace, then re-raise the signal with the default action.
 */
static void crash_handler(int sig, siginfo_t *si, void *uc) {
    (void)uc;
    char b[192];
    int n = snprintf(b, sizeof b, "[kfdr] CRASH sig %d addr %p tid %ld\n", sig, si->si_addr, (long)syscall(SYS_gettid));
    if (write(2, b, n) < 0) { /* nothing to do */ }
    for (int j = 0; j < nmp; j++)
        if ((char *)si->si_addr >= (char *)mps[j].addr && (char *)si->si_addr < (char *)mps[j].addr + mps[j].len) {
            struct bo *o = mps[j].owner;
            n = snprintf(b, sizeof b, "[kfdr]   in model map %p len 0x%zx fd %d off 0x%llx live %d bo flags 0x%08x size 0x%llx\n",
                         mps[j].addr, mps[j].len, mps[j].fd, (unsigned long long)mps[j].off, mps[j].live, o ? o->a.flags : 0,
                         o ? (unsigned long long)o->a.size : 0ULL);
            if (write(2, b, n) < 0) { /* nothing to do */ }
        }
    char comm[64] = "?", pth[64];
    snprintf(pth, sizeof pth, "/proc/self/task/%ld/comm", (long)syscall(SYS_gettid));
    int f = open(pth, O_RDONLY);
    if (f >= 0) {
        int r = read(f, comm, 63);
        if (r > 0) comm[r] = 0;
        close(f);
    }
    if (write(2, "[kfdr]   thread comm ", 21) < 0 || write(2, comm, strlen(comm)) < 0) { /* nothing to do */ }
    void *fr[48];
    int k = backtrace(fr, 48);
    backtrace_symbols_fd(fr, k, 2);
    signal(sig, SIG_DFL);
    raise(sig);
}

__attribute__((constructor)) static void kfdr_crash_init(void) {
    if (!kfdr_env_flag("KFDR_CRASH")) return;
    struct sigaction sa = {0};
    sa.sa_sigaction = crash_handler;
    sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
    int sigs[] = {SIGSEGV, SIGBUS, SIGABRT, SIGILL};
    for (int i = 0; i < 4; i++) sigaction(sigs[i], &sa, NULL);
}
