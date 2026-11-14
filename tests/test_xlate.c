/*
 * Unit tests: ioctl classification and gpu_id / BO-handle translation used when
 * a snapshot is restored on another GPU. No GPU needed.
 */
#include "kfdr.h"
#include "test.h"

#include <string.h>

static const uint32_t OLD = 53277, NEW = 63541, OTHER = 13368;

static void test_classify(void) {
    CHECK(kfdr_classify_kfd(NR_ALLOC_MEMORY_OF_GPU) == 'S');
    CHECK(kfdr_classify_kfd(NR_CREATE_QUEUE) == 'S');
    CHECK(kfdr_classify_kfd(NR_ACQUIRE_VM) == 'P');
    CHECK(kfdr_classify_kfd(0x25) == 'P'); /* RUNTIME_ENABLE */
    CHECK(kfdr_classify_kfd(NR_WAIT_EVENTS) == 'R');
    CHECK(kfdr_classify_kfd(NR_EXPORT_DMABUF) == 'R');
    /* unsupported -> refuse */
    CHECK(kfdr_classify_kfd(0x07) == 0); /* UPDATE_QUEUE */
    CHECK(kfdr_classify_kfd(0x1D) == 0); /* IMPORT_DMABUF */
    CHECK(kfdr_classify_kfd(0x20) == 0); /* SVM */
    CHECK(kfdr_classify_kfd(0x22) == 0); /* CRIU_OP */
    CHECK(kfdr_classify_kfd(0x26) == 0); /* DBG_TRAP */
    CHECK(kfdr_classify_kfd(0x2F) == 0);
    CHECK(kfdr_classify_kfd(0x30) == 0);
    CHECK(kfdr_classify_kfd(0xFF) == 0);

    CHECK(kfdr_drm_nr_allowed(DRM_NR_VERSION));
    CHECK(kfdr_drm_nr_allowed(DRM_NR_AMDGPU_INFO));
    CHECK(kfdr_drm_nr_allowed(DRM_NR_SYNCOBJ_CREATE));
    CHECK(kfdr_drm_nr_allowed(DRM_NR_AMDGPU_GEM_VA));
    CHECK(!kfdr_drm_nr_allowed(0x42)); /* AMDGPU_CTX */
    CHECK(!kfdr_drm_nr_allowed(0x44)); /* AMDGPU_CS */
    CHECK(!kfdr_drm_nr_allowed(0xc1)); /* SYNCOBJ_HANDLE_TO_FD */
    CHECK(!kfdr_drm_nr_allowed(0xc4)); /* SYNCOBJ_RESET */
    CHECK(!kfdr_drm_nr_allowed(0xc5)); /* SYNCOBJ_SIGNAL */
}

static void test_no_translation(void) {
    gpu_old = OLD;
    gpu_new = 0;
    CHECK_EQ_U64(kfdr_tg(OLD), OLD);
    CHECK_EQ_U64(kfdr_th(((uint64_t)OLD << 32) | 5), ((uint64_t)OLD << 32) | 5);
    struct alloc_args a = {.gpu_id = OLD};
    struct xundo u;
    kfdr_xlate(NR_ALLOC_MEMORY_OF_GPU, &a, &u);
    CHECK_EQ_U64(a.gpu_id, OLD);
    CHECK(u.g == NULL && u.h == NULL && u.dp == NULL);
}

static void test_scalars(void) {
    gpu_old = OLD;
    gpu_new = NEW;
    CHECK_EQ_U64(kfdr_tg(OLD), NEW);
    CHECK_EQ_U64(kfdr_tg(OTHER), OTHER);
    CHECK_EQ_U64(kfdr_tg_back(NEW), OLD);
    CHECK_EQ_U64(kfdr_tg_back(OTHER), OTHER);
    CHECK_EQ_U64(kfdr_th(((uint64_t)OLD << 32) | 0x1234), ((uint64_t)NEW << 32) | 0x1234);
    CHECK_EQ_U64(kfdr_th(((uint64_t)OTHER << 32) | 7), ((uint64_t)OTHER << 32) | 7);
    CHECK_EQ_U64(kfdr_th(0), 0);
}

/* gpu_id at a fixed offset of the argument block; translated, then given back. */
static void check_gpu_field(unsigned nr, size_t off) {
    unsigned char buf[128];
    memset(buf, 0xAB, sizeof buf);
    memcpy(buf + off, &OLD, 4);
    unsigned char before[128];
    memcpy(before, buf, sizeof buf);
    struct xundo u;
    kfdr_xlate(nr, buf, &u);
    uint32_t g;
    memcpy(&g, buf + off, 4);
    if (g != NEW) fprintf(stderr, "nr 0x%x: gpu_id at %zu not translated\n", nr, off);
    CHECK_EQ_U64(g, NEW);
    kfdr_xundo_apply(&u);
    CHECK(memcmp(before, buf, sizeof buf) == 0);
}

static void test_inputs(void) {
    gpu_old = OLD;
    gpu_new = NEW;
    check_gpu_field(NR_CREATE_QUEUE, offsetof(struct cq_args, gpu_id));
    check_gpu_field(NR_SET_MEMORY_POLICY, 16);
    check_gpu_field(NR_GET_CLOCK_COUNTERS, 32);
    check_gpu_field(NR_SET_SCRATCH_BACKING_VA, 8);
    check_gpu_field(NR_GET_TILE_CONFIG, 24);
    check_gpu_field(NR_SET_TRAP_HANDLER, 16);
    check_gpu_field(NR_ACQUIRE_VM, 4);
    check_gpu_field(NR_ALLOC_MEMORY_OF_GPU, offsetof(struct alloc_args, gpu_id));
    check_gpu_field(NR_AVAILABLE_MEMORY, 8);

    /* BO handles: FREE / EXPORT_DMABUF at offset 0, CREATE_EVENT page_offset. */
    uint64_t h_old = ((uint64_t)OLD << 32) | 0x42, h_new = ((uint64_t)NEW << 32) | 0x42;
    struct free_args f = {h_old};
    struct xundo u;
    kfdr_xlate(NR_FREE_MEMORY_OF_GPU, &f, &u);
    CHECK_EQ_U64(f.handle, h_new);
    kfdr_xundo_apply(&u);
    CHECK_EQ_U64(f.handle, h_old);

    struct ce_args ce = {.page_offset = h_old};
    kfdr_xlate(NR_CREATE_EVENT, &ce, &u);
    CHECK_EQ_U64(ce.page_offset, h_new);
    kfdr_xundo_apply(&u);
    CHECK_EQ_U64(ce.page_offset, h_old);

    struct ce_args ce0 = {.page_offset = 0}; /* no signal page: untouched */
    kfdr_xlate(NR_CREATE_EVENT, &ce0, &u);
    CHECK_EQ_U64(ce0.page_offset, 0);
    CHECK(u.h == NULL);

    /* Unrelated ioctl: nothing touched. */
    unsigned char buf[64];
    memset(buf, 0x5A, sizeof buf);
    kfdr_xlate(NR_GET_VERSION, buf, &u);
    CHECK(u.g == NULL && u.h == NULL && u.dp == NULL);
}

/* MAP/UNMAP: the caller's device array (a thunk table) must never be modified. */
static void test_map_array(void) {
    gpu_old = OLD;
    gpu_new = NEW;
    uint32_t ids[3] = {OLD, OTHER, OLD};
    uint64_t h_old = ((uint64_t)OLD << 32) | 9;
    struct map_args m = {h_old, (uint64_t)(uintptr_t)ids, 3, 0};
    struct xundo u;
    kfdr_xlate(NR_MAP_MEMORY_TO_GPU, &m, &u);
    uint32_t *seen = (uint32_t *)(uintptr_t)m.dev_ptr;
    CHECK(seen != ids);
    CHECK_EQ_U64(seen[0], NEW);
    CHECK_EQ_U64(seen[1], OTHER);
    CHECK_EQ_U64(seen[2], NEW);
    CHECK_EQ_U64(ids[0], OLD); /* caller's table untouched */
    CHECK_EQ_U64(ids[2], OLD);
    CHECK_EQ_U64(m.handle, ((uint64_t)NEW << 32) | 9);
    kfdr_xundo_apply(&u);
    CHECK_EQ_U64(m.dev_ptr, (uint64_t)(uintptr_t)ids);
    CHECK_EQ_U64(m.handle, h_old);

    /* UNMAP behaves the same. */
    kfdr_xlate(NR_UNMAP_MEMORY_FROM_GPU, &m, &u);
    CHECK((uint32_t *)(uintptr_t)m.dev_ptr != ids);
    kfdr_xundo_apply(&u);
    CHECK_EQ_U64(m.dev_ptr, (uint64_t)(uintptr_t)ids);

    /* No array: pointer left alone. */
    struct map_args m0 = {h_old, 0, 0, 0};
    kfdr_xlate(NR_MAP_MEMORY_TO_GPU, &m0, &u);
    CHECK(u.dp == NULL);
    CHECK_EQ_U64(m0.dev_ptr, 0);
}

/* Values returned by the kernel: new -> old. */
static void test_outputs(void) {
    gpu_old = OLD;
    gpu_new = NEW;
    unsigned char ap[2 * 56];
    memset(ap, 0, sizeof ap);
    memcpy(ap + 48, &NEW, 4);
    memcpy(ap + 56 + 48, &OTHER, 4);
    unsigned char args[16] = {0};
    uint64_t ptr = (uint64_t)(uintptr_t)ap;
    uint32_t n = 2;
    memcpy(args, &ptr, 8);
    memcpy(args + 8, &n, 4);
    kfdr_xlate_out(NR_GET_PROCESS_APERTURES_NEW, args);
    uint32_t g0, g1;
    memcpy(&g0, ap + 48, 4);
    memcpy(&g1, ap + 56 + 48, 4);
    CHECK_EQ_U64(g0, OLD);
    CHECK_EQ_U64(g1, OTHER);

    unsigned char di[32] = {0};
    memcpy(di + 20, &NEW, 4);
    kfdr_xlate_out(NR_GET_DMABUF_INFO, di);
    uint32_t g;
    memcpy(&g, di + 20, 4);
    CHECK_EQ_U64(g, OLD);

    gpu_new = 0; /* no translation configured: untouched */
    memcpy(di + 20, &NEW, 4);
    kfdr_xlate_out(NR_GET_DMABUF_INFO, di);
    memcpy(&g, di + 20, 4);
    CHECK_EQ_U64(g, NEW);
}

int main(void) {
    test_classify();
    test_no_translation();
    test_scalars();
    test_inputs();
    test_map_array();
    test_outputs();
    TEST_DONE("test_xlate");
}
