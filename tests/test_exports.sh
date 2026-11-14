#!/bin/sh
# The shared library must export exactly the public API plus the interposed
# libc functions, and nothing else (internal symbols are hidden).
# usage: tests/test_exports.sh build/libkfdreplay.so
set -eu
lib=${1:-build/libkfdreplay.so}
want="access close fopen fopen64 ioctl kfdr_checkpoint kfdr_drm_report kfdr_last_error kfdr_max_wptr kfdr_restore kfdr_state kfdr_version mmap mmap64 munmap open open64 openat openat64"
got=$(nm -D --defined-only "$lib" | awk '$2 ~ /^[TWiV]$/ {print $3}' | sed 's/@.*//' | sort -u | tr '\n' ' ' | sed 's/ $//')
if [ "$got" != "$want" ]; then
    echo "test_exports: unexpected exported symbols"
    echo "  want: $want"
    echo "  got:  $got"
    exit 1
fi
echo "test_exports: ok"
