/*
 * ioctl classification and gpu_id translation for restore on another GPU.
 *
 * Copyright 2026 The libkfdreplay Authors
 * SPDX-License-Identifier: Apache-2.0
 *
 * KFD gpu_id is a hash identifying one physical device. It appears in ioctl
 * arguments (ALLOC, CREATE_QUEUE, ACQUIRE_VM and the process-level calls), in
 * the device arrays of MAP/UNMAP and in the upper 32 bits of BO handles. When a
 * snapshot is restored on a different GPU (one GPU per process), every such
 * value old -> new is translated, both in our own replay and in the calls ROCr
 * keeps issuing with the old ids it holds in memory. gpu_old is taken from the
 * recorded BOs at checkpoint time, gpu_new from sysfs at restore time.
 *
 * Translation never modifies the caller's memory for good. The ioctl arguments
 * live in thunk (libhsakmt) memory, and the MAP/UNMAP device array *is* one of
 * its tables (all_gpu_id_array / registered_device_id_array). Rewritten in
 * place, those tables would carry the new gpu_id forever, and the "all nodes
 * registered" check in hsakmt_fmm_map_to_gpu_nodes would silently fail
 * (hsa_amd_agents_allow_access -> HSA_STATUS_ERROR_OUT_OF_RESOURCES). So the
 * device array is copied into a scratch buffer and all input fields are put
 * back after the ioctl (kfdr_xundo_apply).
 */
#include "kfdr.h"

#include <string.h>

const char kfd_ioctl_class[KFD_NR_MAX] = {
    [0x01] = 'R', /* GET_VERSION */
    [0x02] = 'S', /* CREATE_QUEUE */
    [0x03] = 'S', /* DESTROY_QUEUE */
    [0x04] = 'P', /* SET_MEMORY_POLICY */
    [0x05] = 'R', /* GET_CLOCK_COUNTERS */
    [0x06] = 'R', /* GET_PROCESS_APERTURES */
    [0x08] = 'S', /* CREATE_EVENT */
    [0x09] = 'S', /* DESTROY_EVENT */
    [0x0A] = 'R', /* SET_EVENT */
    [0x0B] = 'R', /* RESET_EVENT */
    [0x0C] = 'R', /* WAIT_EVENTS */
    [0x11] = 'P', /* SET_SCRATCH_BACKING_VA */
    [0x12] = 'R', /* GET_TILE_CONFIG */
    [0x13] = 'P', /* SET_TRAP_HANDLER */
    [0x14] = 'R', /* GET_PROCESS_APERTURES_NEW */
    [0x15] = 'P', /* ACQUIRE_VM */
    [0x16] = 'S', /* ALLOC_MEMORY_OF_GPU */
    [0x17] = 'S', /* FREE_MEMORY_OF_GPU */
    [0x18] = 'S', /* MAP_MEMORY_TO_GPU */
    [0x19] = 'S', /* UNMAP_MEMORY_FROM_GPU */
    [0x1C] = 'R', /* GET_DMABUF_INFO */
    [0x21] = 'P', /* SET_XNACK_MODE */
    [0x23] = 'R', /* AVAILABLE_MEMORY */
    [0x24] = 'R', /* EXPORT_DMABUF (a dmabuf fd still open at checkpoint is refused separately) */
    [0x25] = 'P', /* RUNTIME_ENABLE */
};

char kfdr_classify_kfd(unsigned nr) { return nr < KFD_NR_MAX ? kfd_ioctl_class[nr] : 0; }

/*
 * DRM is fail-closed: render-node state is not replayed, so any successful DRM
 * ioctl other than the ones below means the kernel holds state the new process
 * would not have (GEM objects, syncobjs, contexts, VA mappings).
 */
int kfdr_drm_nr_allowed(unsigned n) {
    switch (n) {
    /* read-only queries */
    case DRM_NR_VERSION: case DRM_NR_GET_UNIQUE: case DRM_NR_GET_CAP: case DRM_NR_AMDGPU_INFO:
    /* syncobj create/destroy: journaled and replayed */
    case DRM_NR_SYNCOBJ_CREATE: case DRM_NR_SYNCOBJ_DESTROY:
    /* GEM handles / GEM_VA / dmabuf export: tracked as live state, must be zero at checkpoint */
    case DRM_NR_GEM_CLOSE: case DRM_NR_PRIME_H2FD: case DRM_NR_PRIME_FD2H:
    case DRM_NR_AMDGPU_GEM_CREATE: case DRM_NR_AMDGPU_GEM_VA:
    /* state lives on the handle (GEM_MMAP only returns an offset, GEM_METADATA)
       or is not changed (SYNCOBJ_WAIT, SYNCOBJ_TIMELINE_WAIT, SYNCOBJ_QUERY) */
    case DRM_NR_AMDGPU_GEM_MMAP: case DRM_NR_AMDGPU_GEM_METADATA:
    case DRM_NR_SYNCOBJ_WAIT: case DRM_NR_SYNCOBJ_TIMELINE_WAIT: case DRM_NR_SYNCOBJ_QUERY:
        return 1;
    default:
        return 0;
    }
}

uint32_t gpu_old, gpu_new;

uint32_t kfdr_tg(uint32_t g) { return (gpu_new && g == gpu_old) ? gpu_new : g; }

uint64_t kfdr_th(uint64_t h) {
    return (gpu_new && (uint32_t)(h >> 32) == gpu_old) ? (((uint64_t)gpu_new << 32) | (h & 0xffffffffull)) : h;
}

/* new -> old, for values the kernel *returns* (see kfdr_xlate_out). */
uint32_t kfdr_tg_back(uint32_t g) { return (gpu_new && g == gpu_new) ? gpu_old : g; }

static __thread uint32_t xids[256];

/* Translate the input fields of KFD ioctl `nr` old -> new; `u` records how to undo it. */
void kfdr_xlate(unsigned nr, void *arg, struct xundo *u) {
    memset(u, 0, sizeof *u);
    if (!gpu_new || !arg) return;
    unsigned char *b = arg;
    uint32_t *g = NULL;
    uint64_t *h = NULL;
    switch (nr) {
    case NR_CREATE_QUEUE: g = &((struct cq_args *)arg)->gpu_id; break;
    case NR_SET_MEMORY_POLICY: case NR_SET_TRAP_HANDLER: g = (uint32_t *)(b + 16); break;
    case NR_GET_CLOCK_COUNTERS: g = (uint32_t *)(b + 32); break;
    case NR_SET_SCRATCH_BACKING_VA: case NR_AVAILABLE_MEMORY: g = (uint32_t *)(b + 8); break;
    case NR_GET_TILE_CONFIG: g = (uint32_t *)(b + 24); break;
    case NR_ACQUIRE_VM: g = (uint32_t *)(b + 4); break;
    case NR_ALLOC_MEMORY_OF_GPU: g = &((struct alloc_args *)arg)->gpu_id; break;
    case NR_CREATE_EVENT: h = &((struct ce_args *)arg)->page_offset; break;
    case NR_FREE_MEMORY_OF_GPU: case NR_EXPORT_DMABUF: h = (uint64_t *)b; break;
    case NR_MAP_MEMORY_TO_GPU: case NR_UNMAP_MEMORY_FROM_GPU: {
        struct map_args *m = arg;
        h = &m->handle;
        uint32_t *ids = (uint32_t *)(uintptr_t)m->dev_ptr;
        uint32_t n = m->n_devices < 256 ? m->n_devices : 256;
        if (ids && n) {
            for (uint32_t i = 0; i < n; i++) xids[i] = kfdr_tg(ids[i]);
            u->dp = &m->dev_ptr;
            u->dpv = m->dev_ptr;
            m->dev_ptr = (uint64_t)(uintptr_t)xids;
        }
        break;
    }
    }
    if (g) { u->g = g; u->gv = *g; *g = kfdr_tg(*g); }
    if (h && *h) { u->h = h; u->hv = *h; *h = kfdr_th(*h); }
}

/* Give the input fields back to the caller. */
void kfdr_xundo_apply(struct xundo *u) {
    if (u->dp) *u->dp = u->dpv;
    if (u->g) *u->g = u->gv;
    if (u->h) *u->h = u->hv;
}

/*
 * Reverse translation (new -> old) of values the kernel returns: the thunk
 * looks the GPU up by them in its own table (gpu_mem_find_by_gpu_id), which
 * holds the old id (fmm.c: register_graphics_handle, register_shared_memory,
 * init_process_apertures). BO handles carrying the new gpu_id are left alone:
 * they only ever go back into the kernel.
 */
void kfdr_xlate_out(unsigned nr, void *arg) {
    if (!gpu_new || !arg) return;
    unsigned char *b = arg;
    switch (nr) {
    case NR_GET_DMABUF_INFO: { /* kfd_ioctl_get_dmabuf_info_args.gpu_id */
        uint32_t *g = (uint32_t *)(b + 20);
        *g = kfdr_tg_back(*g);
        break;
    }
    case NR_GET_PROCESS_APERTURES_NEW: { /* kfd_process_device_apertures[].gpu_id */
        uint64_t ptr = *(uint64_t *)b;
        uint32_t n = *(uint32_t *)(b + 8);
        for (uint32_t i = 0; ptr && i < n; i++) {
            uint32_t *g = (uint32_t *)(uintptr_t)(ptr + i * 56 + 48);
            *g = kfdr_tg_back(*g);
        }
        break;
    }
    }
}
