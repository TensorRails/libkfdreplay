/*
 * Smoke test of the preloaded library on a host without a GPU: the libc hooks
 * must be transparent, and a checkpoint/restore cycle of a process that never
 * touched the GPU must succeed and walk through the documented states.
 *
 *   LD_PRELOAD=build/libkfdreplay.so build/test_preload
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../include/kfdreplay.h"
#include "test.h"

typedef int (*int_fn)(void);
typedef int (*ckpt_fn)(int);
typedef const char *(*str_fn)(void);

int main(void) {
    /* Looked up at runtime, the way applications use the library. */
    ckpt_fn checkpoint = (ckpt_fn)dlsym(RTLD_DEFAULT, "kfdr_checkpoint");
    int_fn restore = (int_fn)dlsym(RTLD_DEFAULT, "kfdr_restore");
    int_fn state = (int_fn)dlsym(RTLD_DEFAULT, "kfdr_state");
    str_fn version = (str_fn)dlsym(RTLD_DEFAULT, "kfdr_version");
    str_fn last_error = (str_fn)dlsym(RTLD_DEFAULT, "kfdr_last_error");
    if (!checkpoint || !restore || !state || !version || !last_error) {
        fprintf(stderr, "test_preload: libkfdreplay is not preloaded\n");
        return 2;
    }
    CHECK(strcmp(version(), KFDR_VERSION_STRING) == 0);
    CHECK(state() == KFDR_STATE_RUNNING);

    /* libc hooks are transparent */
    char path[] = "/tmp/kfdr-preload-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(write(fd, "hello", 5) == 5);
    close(fd);
    fd = open(path, O_RDONLY);
    CHECK(fd >= 0);
    void *m = mmap(NULL, 4096, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(m != MAP_FAILED && memcmp(m, "hello", 5) == 0);
    CHECK(munmap(m, 4096) == 0);
    close(fd);
    FILE *f = fopen(path, "r");
    char buf[8] = {0};
    CHECK(f && fread(buf, 1, 5, f) == 5 && !strcmp(buf, "hello"));
    if (f) fclose(f);
    CHECK(access(path, R_OK) == 0);
    CHECK(open("/nonexistent/kfdr", O_RDONLY) < 0);
    unlink(path);

    /* empty checkpoint/restore cycle */
    CHECK(checkpoint(1) == 0);
    CHECK(state() == KFDR_STATE_CHECKPOINTED_COLD);
    CHECK(last_error()[0] == 0);
    CHECK(restore() == 0);
    CHECK(state() == KFDR_STATE_RESTORED);
    CHECK(last_error()[0] == 0);
    TEST_DONE("test_preload");
}
