# Releasing `libkfdreplay`

## Cut a release

1. Update `KFDR_VERSION_MAJOR/MINOR/PATCH` and `KFDR_VERSION_STRING` in
   `include/kfdreplay.h`.
2. In `CHANGELOG.md`, add or rename the section heading to
   `## vX.Y.Z — YYYY-MM-DD`. `scripts/release-notes.sh` extracts this section
   for the GitHub Release; missing or empty release notes fail the workflow.
3. Run the GPU acceptance suite on an Instinct pod (`examples/hip_toy`,
   `examples/pytorch` including `CHAIN=3` and `NEG=1`, `examples/vllm_vmm`,
   `examples/vllm_offline`, `examples/vllm_serve`).
4. Commit, tag and push:

   ```bash
   git tag -a vX.Y.Z -m vX.Y.Z
   git push origin main vX.Y.Z
   ```

`.github/workflows/release.yml` builds `libkfdreplay.so` in a pinned Debian 12
container (`glibc >= 2.34`), verifies symbol exports, GLIBC baseline and header
version alignment, tests the exact packaged `.so` on Ubuntu 22.04 and Ubuntu
24.04 (including a real DMTCP cold cycle), records build provenance when the
repository is public, and publishes:

| Asset | Contents |
|---|---|
| `libkfdreplay.so` | Prebuilt x86-64 shared library (`SONAME` `libkfdreplay.so`) |
| `libkfdreplay-vX.Y.Z-linux-amd64.tar.gz` | `libkfdreplay.so`, `include/kfdreplay.h`, `BUILDINFO`, `LICENSE`, `NOTICE`, `README.md` |
| `libkfdreplay-vX.Y.Z-source.tar.gz` | Tagged git source archive |
| `SHA256SUMS` | SHA-256 checksums of all release assets |

Tags containing a pre-release suffix (`-rc.N`, `-alpha.N`, `-beta.N`) are
published as GitHub pre-releases.

## Verify a release

```bash
V=v0.1.0
sha256sum --ignore-missing -c SHA256SUMS
```

Releases published once the repository is public also carry GitHub build
provenance (`gh attestation verify libkfdreplay-$V-linux-amd64.tar.gz --repo TensorRails/libkfdreplay`).

## Snapshot compatibility

DMTCP maps `libkfdreplay.so` from its file on restart: donor and receiver must
have a byte-identical `libkfdreplay.so` at the same path with the same size and
mtime. Install each version into its own directory (for example
`/opt/warmctl/kfdreplay/v0.1.0/libkfdreplay.so`) rather than overwriting a
library referenced by existing snapshots.
