#!/usr/bin/env python3
"""Drop files from the page cache without privileges (posix_fadvise DONTNEED),
so that the next dmtcp_restart reads the checkpoint images from disk."""
import os
import sys

for path in sys.argv[1:]:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
