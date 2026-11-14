/*
 * Logging, diagnostic switches and the last-error buffer.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Diagnostic switches are read from the environment. Note that a process
 * restored by DMTCP keeps the environment it had at dmtcp_launch: variables set
 * for dmtcp_restart do not reach it. Diagnostics that must be switched on at
 * restore time therefore also look at marker files (KFDR_TRACE_FILE,
 * KFDR_PAUSE_FILE) on the restore host.
 */
#include "kfdr.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

double kfdr_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int kfdr_env_flag(const char *name) {
    const char *v = getenv(name);
    return v && *v == '1';
}

int kfdr_debug_on(void) { return kfdr_env_flag("KFDR_DEBUG"); }

static int trace_cached;
/* Evaluated once, when a cold restore completes. */
void kfdr_trace_refresh(void) {
    trace_cached = kfdr_env_flag("KFDR_TRACE") || access(KFDR_TRACE_FILE, F_OK) == 0;
}
int kfdr_trace_on(void) { return restored_cold && trace_cached; }

static char last_error[512];

void kfdr_clear_error(void) { last_error[0] = 0; }

void kfdr_fail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(last_error, sizeof last_error, fmt, ap);
    va_end(ap);
    fprintf(stderr, "[kfdr] %s\n", last_error);
}

KFDR_API const char *kfdr_last_error(void) { return last_error; }
