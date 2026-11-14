/*
 * ABI check: the KFD/DRM structures, ioctl numbers and field offsets declared
 * in src/kfdr.h (the library does not include kernel headers) must match the
 * kernel UAPI headers. Skipped (with a message) when the headers are missing,
 * e.g. on non-Linux hosts. Debian/Ubuntu: apt install linux-libc-dev.
 */
#include "kfdr.h"
#include "test.h"

#if defined(__has_include)
#if __has_include(<linux/kfd_ioctl.h>) && __has_include(<drm/drm.h>) && __has_include(<drm/amdgpu_drm.h>)
#define HAVE_UAPI 1
#endif
#endif

#ifdef HAVE_UAPI
#include <drm/amdgpu_drm.h>
#include <drm/drm.h>
#include <linux/kfd_ioctl.h>

#define SAME_SIZE(ours, theirs) _Static_assert(sizeof(ours) == sizeof(theirs), #ours " size differs from " #theirs)
#define SAME_OFF(ours, of, theirs, tf) \
    _Static_assert(offsetof(ours, of) == offsetof(theirs, tf), #ours "." #of " offset differs from " #theirs "." #tf)

SAME_SIZE(struct alloc_args, struct kfd_ioctl_alloc_memory_of_gpu_args);
SAME_OFF(struct alloc_args, va_addr, struct kfd_ioctl_alloc_memory_of_gpu_args, va_addr);
SAME_OFF(struct alloc_args, size, struct kfd_ioctl_alloc_memory_of_gpu_args, size);
SAME_OFF(struct alloc_args, handle, struct kfd_ioctl_alloc_memory_of_gpu_args, handle);
SAME_OFF(struct alloc_args, mmap_offset, struct kfd_ioctl_alloc_memory_of_gpu_args, mmap_offset);
SAME_OFF(struct alloc_args, gpu_id, struct kfd_ioctl_alloc_memory_of_gpu_args, gpu_id);
SAME_OFF(struct alloc_args, flags, struct kfd_ioctl_alloc_memory_of_gpu_args, flags);

SAME_SIZE(struct free_args, struct kfd_ioctl_free_memory_of_gpu_args);

SAME_SIZE(struct map_args, struct kfd_ioctl_map_memory_to_gpu_args);
SAME_SIZE(struct map_args, struct kfd_ioctl_unmap_memory_from_gpu_args);
SAME_OFF(struct map_args, handle, struct kfd_ioctl_map_memory_to_gpu_args, handle);
SAME_OFF(struct map_args, dev_ptr, struct kfd_ioctl_map_memory_to_gpu_args, device_ids_array_ptr);
SAME_OFF(struct map_args, n_devices, struct kfd_ioctl_map_memory_to_gpu_args, n_devices);
SAME_OFF(struct map_args, n_success, struct kfd_ioctl_map_memory_to_gpu_args, n_success);

/* sdma_engine_id (+pad) was appended in newer KFD ABI versions (amdgpu DKMS
   shipped with ROCm 6.x+). Older kernel headers end at ctl_stack_size (88 bytes);
   KFD accepts both sizes. The library uses the newer 96-byte layout. */
_Static_assert(sizeof(struct cq_args) == 96, "cq_args size");
_Static_assert(sizeof(struct kfd_ioctl_create_queue_args) == 96 || sizeof(struct kfd_ioctl_create_queue_args) == 88,
               "unexpected kfd_ioctl_create_queue_args size");
SAME_OFF(struct cq_args, ring, struct kfd_ioctl_create_queue_args, ring_base_address);
SAME_OFF(struct cq_args, wptr, struct kfd_ioctl_create_queue_args, write_pointer_address);
SAME_OFF(struct cq_args, rptr, struct kfd_ioctl_create_queue_args, read_pointer_address);
SAME_OFF(struct cq_args, doorbell_offset, struct kfd_ioctl_create_queue_args, doorbell_offset);
SAME_OFF(struct cq_args, ring_size, struct kfd_ioctl_create_queue_args, ring_size);
SAME_OFF(struct cq_args, gpu_id, struct kfd_ioctl_create_queue_args, gpu_id);
SAME_OFF(struct cq_args, queue_type, struct kfd_ioctl_create_queue_args, queue_type);
SAME_OFF(struct cq_args, queue_id, struct kfd_ioctl_create_queue_args, queue_id);
SAME_OFF(struct cq_args, eop, struct kfd_ioctl_create_queue_args, eop_buffer_address);
SAME_OFF(struct cq_args, ctx_save, struct kfd_ioctl_create_queue_args, ctx_save_restore_address);
SAME_OFF(struct cq_args, ctl_stack_size, struct kfd_ioctl_create_queue_args, ctl_stack_size);

SAME_SIZE(struct dq_args, struct kfd_ioctl_destroy_queue_args);

SAME_SIZE(struct ce_args, struct kfd_ioctl_create_event_args);
SAME_OFF(struct ce_args, page_offset, struct kfd_ioctl_create_event_args, event_page_offset);
SAME_OFF(struct ce_args, trigger, struct kfd_ioctl_create_event_args, event_trigger_data);
SAME_OFF(struct ce_args, type, struct kfd_ioctl_create_event_args, event_type);
SAME_OFF(struct ce_args, node_id, struct kfd_ioctl_create_event_args, node_id);
SAME_OFF(struct ce_args, event_id, struct kfd_ioctl_create_event_args, event_id);
SAME_OFF(struct ce_args, slot, struct kfd_ioctl_create_event_args, event_slot_index);

SAME_SIZE(struct ev_args, struct kfd_ioctl_destroy_event_args);
SAME_SIZE(struct ev_args, struct kfd_ioctl_set_event_args);

/* gpu_id offsets hard-coded in kfdr_xlate() / kfdr_xlate_out(). */
_Static_assert(offsetof(struct kfd_ioctl_set_memory_policy_args, gpu_id) == 16, "SET_MEMORY_POLICY.gpu_id");
_Static_assert(offsetof(struct kfd_ioctl_get_clock_counters_args, gpu_id) == 32, "GET_CLOCK_COUNTERS.gpu_id");
_Static_assert(offsetof(struct kfd_ioctl_set_scratch_backing_va_args, gpu_id) == 8, "SET_SCRATCH_BACKING_VA.gpu_id");
_Static_assert(offsetof(struct kfd_ioctl_get_tile_config_args, gpu_id) == 24, "GET_TILE_CONFIG.gpu_id");
_Static_assert(offsetof(struct kfd_ioctl_set_trap_handler_args, gpu_id) == 16, "SET_TRAP_HANDLER.gpu_id");
_Static_assert(offsetof(struct kfd_ioctl_acquire_vm_args, gpu_id) == 4, "ACQUIRE_VM.gpu_id");
_Static_assert(offsetof(struct kfd_ioctl_get_dmabuf_info_args, gpu_id) == 20, "GET_DMABUF_INFO.gpu_id");
_Static_assert(offsetof(struct kfd_ioctl_get_process_apertures_new_args, kfd_process_device_apertures_ptr) == 0, "APERTURES_NEW.ptr");
_Static_assert(offsetof(struct kfd_ioctl_get_process_apertures_new_args, num_of_nodes) == 8, "APERTURES_NEW.num_of_nodes");
_Static_assert(sizeof(struct kfd_process_device_apertures) == 56, "kfd_process_device_apertures size");
_Static_assert(offsetof(struct kfd_process_device_apertures, gpu_id) == 48, "kfd_process_device_apertures.gpu_id");
_Static_assert(offsetof(struct kfd_ioctl_free_memory_of_gpu_args, handle) == 0, "FREE.handle");
#ifdef AMDKFD_IOC_AVAILABLE_MEMORY
_Static_assert(offsetof(struct kfd_ioctl_get_available_memory_args, gpu_id) == 8, "AVAILABLE_MEMORY.gpu_id");
#endif
#ifdef AMDKFD_IOC_EXPORT_DMABUF
_Static_assert(offsetof(struct kfd_ioctl_export_dmabuf_args, handle) == 0, "EXPORT_DMABUF.handle");
#endif

/* DRM fields read by the hooks. */
_Static_assert(offsetof(struct drm_syncobj_create, handle) == 0, "drm_syncobj_create.handle");
_Static_assert(offsetof(struct drm_syncobj_create, flags) == 4, "drm_syncobj_create.flags");
_Static_assert(offsetof(struct drm_syncobj_destroy, handle) == 0, "drm_syncobj_destroy.handle");
_Static_assert(sizeof(struct drm_syncobj_create) == 8 && sizeof(struct drm_syncobj_destroy) == 8, "syncobj arg size");
_Static_assert(offsetof(struct drm_gem_close, handle) == 0, "drm_gem_close.handle");
_Static_assert(offsetof(struct drm_prime_handle, handle) == 0, "drm_prime_handle.handle");
_Static_assert(offsetof(union drm_amdgpu_gem_create, out.handle) == 0, "drm_amdgpu_gem_create.out.handle");
_Static_assert(offsetof(struct drm_amdgpu_gem_va, operation) == 8, "drm_amdgpu_gem_va.operation");

static void check_nr(const char *name, unsigned long req, unsigned ours) {
    if ((unsigned)_IOC_NR(req) != ours || _IOC_TYPE(req) == 0) {
        fprintf(stderr, "%s: _IOC_NR 0x%x, ours 0x%x\n", name, (unsigned)_IOC_NR(req), ours);
        test_failures++;
    }
}
#define NR(name, ours) check_nr(#name, name, ours)

int main(void) {
    /* Full request codes (dir | size | type | nr) the library builds itself. */
    if (sizeof(struct kfd_ioctl_create_queue_args) == sizeof(struct cq_args))
        CHECK_EQ_U64(IOC_CREATE_QUEUE, AMDKFD_IOC_CREATE_QUEUE);
    else /* older header: same request apart from the size field */
        CHECK_EQ_U64(IOC_CREATE_QUEUE & ~((unsigned long)_IOC_SIZEMASK << _IOC_SIZESHIFT),
                     AMDKFD_IOC_CREATE_QUEUE & ~((unsigned long)_IOC_SIZEMASK << _IOC_SIZESHIFT));
    CHECK_EQ_U64(IOC_DESTROY_QUEUE, AMDKFD_IOC_DESTROY_QUEUE);
    CHECK_EQ_U64(IOC_CREATE_EVENT, AMDKFD_IOC_CREATE_EVENT);
    CHECK_EQ_U64(IOC_DESTROY_EVENT, AMDKFD_IOC_DESTROY_EVENT);
    CHECK_EQ_U64(IOC_SET_EVENT, AMDKFD_IOC_SET_EVENT);
    CHECK_EQ_U64(IOC_ALLOC, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU);
    CHECK_EQ_U64(IOC_FREE, AMDKFD_IOC_FREE_MEMORY_OF_GPU);
    CHECK_EQ_U64(IOC_MAP, AMDKFD_IOC_MAP_MEMORY_TO_GPU);
    CHECK_EQ_U64(IOC_UNMAP, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU);
    CHECK_EQ_U64(_IOWR('d', DRM_NR_SYNCOBJ_CREATE, uint32_t[2]), DRM_IOCTL_SYNCOBJ_CREATE);
    CHECK_EQ_U64(_IOWR('d', DRM_NR_SYNCOBJ_DESTROY, uint32_t[2]), DRM_IOCTL_SYNCOBJ_DESTROY);

    /* ioctl numbers referenced by name. */
    NR(AMDKFD_IOC_GET_VERSION, NR_GET_VERSION);
    NR(AMDKFD_IOC_SET_MEMORY_POLICY, NR_SET_MEMORY_POLICY);
    NR(AMDKFD_IOC_GET_CLOCK_COUNTERS, NR_GET_CLOCK_COUNTERS);
    NR(AMDKFD_IOC_WAIT_EVENTS, NR_WAIT_EVENTS);
    NR(AMDKFD_IOC_SET_SCRATCH_BACKING_VA, NR_SET_SCRATCH_BACKING_VA);
    NR(AMDKFD_IOC_GET_TILE_CONFIG, NR_GET_TILE_CONFIG);
    NR(AMDKFD_IOC_SET_TRAP_HANDLER, NR_SET_TRAP_HANDLER);
    NR(AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, NR_GET_PROCESS_APERTURES_NEW);
    NR(AMDKFD_IOC_ACQUIRE_VM, NR_ACQUIRE_VM);
    NR(AMDKFD_IOC_GET_DMABUF_INFO, NR_GET_DMABUF_INFO);
#ifdef AMDKFD_IOC_AVAILABLE_MEMORY
    NR(AMDKFD_IOC_AVAILABLE_MEMORY, NR_AVAILABLE_MEMORY);
#endif
#ifdef AMDKFD_IOC_EXPORT_DMABUF
    NR(AMDKFD_IOC_EXPORT_DMABUF, NR_EXPORT_DMABUF);
#endif

    /* Classification table vs. the kernel's ioctl list. */
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_SET_XNACK_MODE)) == 'P');
#ifdef AMDKFD_IOC_RUNTIME_ENABLE
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_RUNTIME_ENABLE)) == 'P');
#endif
#ifdef AMDKFD_IOC_CRIU_OP
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_CRIU_OP)) == 0);
#endif
#ifdef AMDKFD_IOC_SVM
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_SVM)) == 0);
#endif
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_IMPORT_DMABUF)) == 0);
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_UPDATE_QUEUE)) == 0);
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_SET_CU_MASK)) == 0);
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_ALLOC_QUEUE_GWS)) == 0);
    CHECK(kfdr_classify_kfd(_IOC_NR(AMDKFD_IOC_SMI_EVENTS)) == 0);
    CHECK(AMDKFD_COMMAND_END <= KFD_NR_MAX);

    /* Flags. */
    CHECK_EQ_U64(F_VRAM, KFD_IOC_ALLOC_MEM_FLAGS_VRAM);
    CHECK_EQ_U64(F_GTT, KFD_IOC_ALLOC_MEM_FLAGS_GTT);
    CHECK_EQ_U64(F_USERPTR, KFD_IOC_ALLOC_MEM_FLAGS_USERPTR);
    CHECK_EQ_U64(F_DOORBELL, KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL);
    CHECK_EQ_U64(F_MMIO, KFD_IOC_ALLOC_MEM_FLAGS_MMIO_REMAP);
    CHECK_EQ_U64(F_WRITABLE, (uint32_t)KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE);
    CHECK_EQ_U64(QUEUE_TYPE_COMPUTE_AQL, KFD_IOC_QUEUE_TYPE_COMPUTE_AQL);

    /* DRM numbers. */
    NR(DRM_IOCTL_VERSION, DRM_NR_VERSION);
    NR(DRM_IOCTL_GET_UNIQUE, DRM_NR_GET_UNIQUE);
    NR(DRM_IOCTL_GEM_CLOSE, DRM_NR_GEM_CLOSE);
    NR(DRM_IOCTL_GET_CAP, DRM_NR_GET_CAP);
    NR(DRM_IOCTL_PRIME_HANDLE_TO_FD, DRM_NR_PRIME_H2FD);
    NR(DRM_IOCTL_PRIME_FD_TO_HANDLE, DRM_NR_PRIME_FD2H);
    NR(DRM_IOCTL_AMDGPU_GEM_CREATE, DRM_NR_AMDGPU_GEM_CREATE);
    NR(DRM_IOCTL_AMDGPU_GEM_MMAP, DRM_NR_AMDGPU_GEM_MMAP);
    NR(DRM_IOCTL_AMDGPU_INFO, DRM_NR_AMDGPU_INFO);
    NR(DRM_IOCTL_AMDGPU_GEM_METADATA, DRM_NR_AMDGPU_GEM_METADATA);
    NR(DRM_IOCTL_AMDGPU_GEM_VA, DRM_NR_AMDGPU_GEM_VA);
    NR(DRM_IOCTL_SYNCOBJ_WAIT, DRM_NR_SYNCOBJ_WAIT);
    NR(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, DRM_NR_SYNCOBJ_TIMELINE_WAIT);
    NR(DRM_IOCTL_SYNCOBJ_QUERY, DRM_NR_SYNCOBJ_QUERY);
    CHECK_EQ_U64(AMDGPU_VA_OP_MAP, 1);
    CHECK_EQ_U64(AMDGPU_VA_OP_UNMAP, 2);
    /* Not on the DRM allow-list: state that would not survive a cold restore. */
    CHECK(!kfdr_drm_nr_allowed(_IOC_NR(DRM_IOCTL_AMDGPU_CTX)));
    CHECK(!kfdr_drm_nr_allowed(_IOC_NR(DRM_IOCTL_AMDGPU_CS)));
    CHECK(!kfdr_drm_nr_allowed(_IOC_NR(DRM_IOCTL_SYNCOBJ_SIGNAL)));
    CHECK(!kfdr_drm_nr_allowed(_IOC_NR(DRM_IOCTL_SYNCOBJ_RESET)));
    CHECK(!kfdr_drm_nr_allowed(_IOC_NR(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD)));
    CHECK(!kfdr_drm_nr_allowed(_IOC_NR(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE)));
    TEST_DONE("test_abi");
}

#else
int main(void) {
    printf("test_abi: skipped (kernel UAPI headers <linux/kfd_ioctl.h>, <drm/*.h> not found)\n");
    return 0;
}
#endif
