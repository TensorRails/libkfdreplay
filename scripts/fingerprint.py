#!/usr/bin/env python3
"""Print a JSON fingerprint of the ROCm stack actually loaded by PyTorch (paths,
SHA-256 and build-id of runtime libraries, driver, GPU). Attach it to bug reports."""
import hashlib
import json
import os
import platform
import subprocess
import sys

import torch

torch.zeros(1, device="cuda")
torch.cuda.synchronize()
(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) @ torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)).sum().item()
libs = sorted({l.split()[-1] for l in open("/proc/self/maps") if ".so" in l and l.split()[-1].startswith("/")})
want = ("hsa-runtime", "amdhip", "hsakmt", "drm", "rocblas", "hipblaslt", "rccl", "torch_hip", "c10_hip", "comgr", "kfdreplay")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def buildid(p):
    try:
        return subprocess.run(["readelf", "-n", p], capture_output=True, text=True).stdout.split("Build ID:")[1].split()[0]
    except Exception:
        return None


fp = {"kernel": platform.release(), "python": sys.version.split()[0], "torch": torch.__version__, "torch_hip": torch.version.hip,
      "gpu": torch.cuda.get_device_name(0), "gcn_arch": torch.cuda.get_device_properties(0).gcnArchName,
      "env": {k: v for k, v in os.environ.items() if k.startswith(("HSA_", "HIP_", "ROCR_", "LD_", "PYTORCH_", "AMD_", "KFDR_"))},
      "libs": {p: {"sha256": sha(p), "build_id": buildid(p)} for p in libs if any(w in p for w in want)}}
for f in ("/sys/module/amdgpu/version", "/proc/driver/amdgpu/version"):
    if os.path.exists(f):
        fp["amdgpu_module"] = open(f).read().strip()
print(json.dumps(fp, indent=1))
