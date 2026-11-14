/*
 * KFD sysfs topology: which GPU this container owns, and path rewriting for
 * restore on another GPU.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kfdr.h"

#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Overridable only by the unit tests. */
const char *kfdr_topology_dir = "/sys/class/kfd/kfd/topology/nodes";
const char *kfdr_dri_dir = "/dev/dri";

int topo_old = -1, topo_new = -1;

/*
 * ROCr/thunk know their GPU by the topology node number of the source host and
 * may read sysfs by that number after restore. In the new container that node
 * is a different GPU (or not readable at all). Paths ".../topology/nodes/<old>/..."
 * are rewritten to ".../nodes/<new>/...". Returns `p` unchanged or `buf`.
 *
 * (On the tested stack ROCr did not read sysfs after restore, so this is a
 * safety net rather than a requirement.)
 */
const char *kfdr_topo_xlate(const char *p, char *buf, size_t n) {
    if (!p || topo_new < 0 || topo_old == topo_new) return p;
    const char *k = strstr(p, "/topology/nodes/");
    if (!k) return p;
    const char *num = k + 16;
    char *end;
    long v = strtol(num, &end, 10);
    if (end == num || v != topo_old || (*end != '/' && *end != 0)) return p;
    snprintf(buf, n, "%.*s%d%s", (int)(num - p), p, topo_new, end);
    static int nl;
    if (nl++ < 5) kfdr_dbg("topology path %s -> %s\n", p, buf);
    return buf;
}

/* Topology node whose gpu_id == g (a container can read its own node). -1 if none. */
int kfdr_topo_node_of(uint32_t g) {
    /* Bypass our own fopen hook: no path rewriting here. */
    static FILE *(*rf)(const char *, const char *);
    if (!rf) rf = dlsym(RTLD_NEXT, "fopen");
    for (int n = 0; n < 128; n++) {
        char q[512];
        unsigned v = 0;
        snprintf(q, sizeof q, "%s/%d/gpu_id", kfdr_topology_dir, n);
        FILE *f = rf(q, "r");
        if (!f) {
            if (n > 16 && errno == ENOENT) break;
            continue;
        }
        int ok = fscanf(f, "%u", &v) == 1;
        fclose(f);
        if (ok && v == g) return n;
    }
    return -1;
}

/*
 * gpu_id of the GPU of this container: the topology node whose
 * drm_render_minor exists under /dev/dri (topology may list every GPU of the
 * host while only one render node is passed into the container).
 * Returns 0 if there is not exactly one such GPU (restore on another GPU is
 * then not attempted).
 */
uint32_t kfdr_sysfs_gpu_id(void) {
    uint32_t found = 0;
    int nfound = 0;
    for (int n = 0; n < 128; n++) {
        char p[512];
        unsigned v = 0;
        int minor = -1;
        snprintf(p, sizeof p, "%s/%d/gpu_id", kfdr_topology_dir, n);
        FILE *f = fopen(p, "r");
        if (!f) {
            if (n > 16) break;
            continue;
        }
        if (fscanf(f, "%u", &v) != 1) v = 0;
        fclose(f);
        if (!v) continue; /* CPU node */
        snprintf(p, sizeof p, "%s/%d/properties", kfdr_topology_dir, n);
        if ((f = fopen(p, "r"))) {
            char k[64];
            long long x;
            while (fscanf(f, "%63s %lld", k, &x) == 2)
                if (!strcmp(k, "drm_render_minor")) minor = (int)x;
            fclose(f);
        }
        snprintf(p, sizeof p, "%s/renderD%d", kfdr_dri_dir, minor);
        if (minor >= 0 && access(p, F_OK) == 0) {
            found = v;
            nfound++;
        }
    }
    return nfound == 1 ? found : 0;
}
