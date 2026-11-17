/* Recorder boundary tests through the ioctl/mmap hooks and checkpoint API.
   A memfd with mocked successful KFD ioctls stands in for the device. */
#include "kfdr.h"
#include "test.h"

#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

static int successful_ioctl(int fd, unsigned long req, ...) { return 0; }

static void expect_refusal(int fd, const char *message) {
    for (int cold = 0; cold <= 1; cold++) {
        CHECK(kfdr_checkpoint(cold) == -1);
        CHECK(strstr(kfdr_last_error(), message) != NULL);
        CHECK(kfdr_state() == KFDR_STATE_RUNNING);
        CHECK(fcntl(fd, F_GETFD) >= 0);
    }
}

static void event_churn(int fd) {
    struct ce_args create = {.event_id = 1};
    struct ev_args destroy = {.event_id = 1};
    for (int i = 0; i < MAX_EVENT; i++) {
        CHECK(ioctl(fd, IOC_CREATE_EVENT, &create) == 0);
        CHECK(ioctl(fd, IOC_DESTROY_EVENT, &destroy) == 0);
    }
    /* All calls through the last slot were recorded. An empty warm cycle
       succeeds and the application can create another event afterwards. */
    CHECK(nev == MAX_EVENT);
    CHECK(kfdr_checkpoint(0) == 0);
    CHECK(kfdr_restore() == 0);
    fdkind[fd] = FD_KFD; /* checkpoint reclassified the real memfd */
    CHECK(ioctl(fd, IOC_CREATE_EVENT, &create) == 0);
    expect_refusal(fd, "event recorder capacity");
    CHECK(ioctl(fd, IOC_DESTROY_EVENT, &destroy) == 0);
    expect_refusal(fd, "event recorder capacity");
}

static void buffers(int fd) {
    struct alloc_args alloc = {.handle = 1};
    struct free_args release = {.handle = 1};
    nbo = MAX_BO - 1;
    CHECK(ioctl(fd, IOC_ALLOC, &alloc) == 0);
    CHECK(ioctl(fd, IOC_FREE, &release) == 0);
    CHECK(nbo == MAX_BO && !record_overflow);
    CHECK(ioctl(fd, IOC_ALLOC, &alloc) == 0);
    expect_refusal(fd, "buffer object recorder capacity");
}

static void queues(int fd) {
    struct cq_args create = {.queue_id = 1};
    struct dq_args destroy = {.queue_id = 1};
    nq = MAX_QUEUE - 1;
    CHECK(ioctl(fd, IOC_CREATE_QUEUE, &create) == 0);
    CHECK(ioctl(fd, IOC_DESTROY_QUEUE, &destroy) == 0);
    CHECK(nq == MAX_QUEUE && !record_overflow);
    CHECK(ioctl(fd, IOC_CREATE_QUEUE, &create) == 0);
    expect_refusal(fd, "queue recorder capacity");
}

static void mappings(int fd) {
    nmp = MAX_MAP - 1;
    CHECK(ftruncate(fd, 4096) == 0);
    void *p = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(p != MAP_FAILED && nmp == MAX_MAP && !record_overflow);
    CHECK(munmap(p, 4096) == 0);
    p = mmap(NULL, 4096, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(p != MAP_FAILED);
    expect_refusal(fd, "device mapping recorder capacity");
    CHECK(munmap(p, 4096) == 0);
}

static int copy_fd, copy_attempts;
static void *(*libc_mmap)(void *, size_t, int, int, int, off_t);

/* Model a driver allocation during the checkpoint's own buffer copy. The
   initial preflight has already run; teardown must still refuse and unfreeze. */
static void *map_during_copy(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (fd == -1 && flags == MAP_SHARED) {
        struct ce_args event = {.event_id = 1};
        fdkind[copy_fd] = FD_KFD;
        CHECK(ioctl(copy_fd, IOC_CREATE_EVENT, &event) == 0);
        copy_attempts++;
        return libc_mmap(NULL, len, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    return libc_mmap(addr, len, prot, flags, fd, off);
}

static void overflow_during_checkpoint(int fd) {
    struct alloc_args buffer = {.handle = 1, .size = 4096, .flags = F_VRAM};
    CHECK(ioctl(fd, IOC_ALLOC, &buffer) == 0);
    /* kfdr_checkpoint(0) copies non-signal VRAM through real_mmap. */
    setenv("KFDR_COPY", "cpu", 1);
    copy_fd = fd;
    libc_mmap = real_mmap;
    real_mmap = map_during_copy;
    nev = MAX_EVENT;
    expect_refusal(fd, "event recorder capacity");
    CHECK(copy_attempts == 1);
    CHECK(bos[0].data == NULL);
}

int main(void) {
    /* Each scenario starts with fresh process-local recording state. */
    void (*cases[])(int) = {event_churn, buffers, queues, mappings, overflow_during_checkpoint};
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        pid_t pid = fork();
        CHECK(pid >= 0);
        if (pid == 0) {
            kfdr_resolve_real();
            real_ioctl = successful_ioctl;
            int fd = memfd_create("kfdr-capacity", MFD_CLOEXEC);
            if (fd < 0 || fd >= MAX_DEV_FD) _exit(2);
            fdkind[fd] = FD_KFD;
            cases[i](fd);
            _exit(test_failures ? 1 : 0);
        }
        if (pid > 0) {
            int status;
            CHECK(waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0);
        }
    }
    TEST_DONE("test_capacity");
}
