# libkfdreplay — build with plain gcc; no ROCm SDK required.
#
#   make            build build/libkfdreplay.so
#   make test       build and run the GPU-free unit tests
#   make install    install into $(PREFIX) (default /usr/local)
#   make clean
#
# Only libc, libdl and libpthread are linked. The KFD/DRM structures the
# library needs are declared in src/kfdr.h; tests/test_abi.c checks them
# against <linux/kfd_ioctl.h> and <drm/*.h> when those headers are installed
# (Debian/Ubuntu: linux-libc-dev).

CC      ?= gcc
CFLAGS  ?= -O2 -g
CFLAGS  += -std=gnu11 -fPIC -fvisibility=hidden -D_GNU_SOURCE -Wall -Wextra -Wno-unused-parameter
LDLIBS  := -ldl -lpthread
PREFIX  ?= /usr/local

BUILD   := build
SRCS    := $(wildcard src/*.c)
OBJS    := $(SRCS:src/%.c=$(BUILD)/%.o)
LIB     := $(BUILD)/libkfdreplay.so

# Objects the unit tests link directly (no libc interposition).
TEST_OBJS := $(BUILD)/state.o $(BUILD)/log.o $(BUILD)/xlate.o $(BUILD)/topology.o
TESTS     := $(BUILD)/test_xlate $(BUILD)/test_topology $(BUILD)/test_abi

.PHONY: all test test-dmtcp install clean
all: $(LIB)

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: src/%.c src/kfdr.h include/kfdreplay.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(LIB): $(OBJS)
	$(CC) $(CFLAGS) -shared -Wl,-soname,libkfdreplay.so -o $@ $^ $(LDLIBS)

$(BUILD)/test_%: tests/test_%.c tests/test.h $(TEST_OBJS)
	$(CC) $(CFLAGS) -Isrc -o $@ $< $(TEST_OBJS) $(LDLIBS)

# Runs with the library preloaded; does not link the library objects.
$(BUILD)/test_preload: tests/test_preload.c tests/test.h include/kfdreplay.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< -ldl

test: $(LIB) $(TESTS) $(BUILD)/test_preload
	@set -e; for t in $(TESTS); do $$t; done
	@LD_PRELOAD=$(abspath $(LIB)) $(BUILD)/test_preload 2>$(BUILD)/test_preload.stderr || { cat $(BUILD)/test_preload.stderr; exit 1; }
	@LD_PRELOAD=$(abspath $(LIB)) sh -c 'ls / >/dev/null && cat /proc/self/status >/dev/null' && echo "preloaded shell: ok"
	@sh tests/test_exports.sh $(LIB)

test-dmtcp: $(LIB)
	@bash tests/test_cold_cycle.sh

install: $(LIB)
	install -d $(DESTDIR)$(PREFIX)/lib $(DESTDIR)$(PREFIX)/include
	install -m 0755 $(LIB) $(DESTDIR)$(PREFIX)/lib/libkfdreplay.so
	install -m 0644 include/kfdreplay.h $(DESTDIR)$(PREFIX)/include/kfdreplay.h

clean:
	rm -rf $(BUILD)
