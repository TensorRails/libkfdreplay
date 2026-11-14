# HIP VMM (vLLM's CuMemAllocator pool, the path sleep mode uses) survives a cold checkpoint/restore.
#
#   (under dmtcp_launch, see run.sh) python vmm_pool.py <ckdir>
# Before the snapshot: the "weights" pool holds a tensor with known contents, the "kv" pool a work buffer;
# allocator.sleep(offload_tags=("weights",)) moves weights to the CPU and releases the physical memory of
# both pools (unmap/release); the VA stays reserved. No live DRM/VMM state remains -> cold checkpoint.
# After restart: kfdr_restore -> allocator.wake_up() (create/map at the old VAs) -> weights checksum,
# kv write/read, a kernel on both, a new pool allocation, a second sleep/wake in the restored process.
import ctypes, hashlib, json, os, sys, time
import torch
from vllm.device_allocator.cumem import CuMemAllocator

ckdir = sys.argv[1]
lib = ctypes.CDLL(None)
out = {"test_id": "vllm-vmm-pool-cold"}
alloc = CuMemAllocator.get_instance()
GiB = 1 << 30
free0 = torch.cuda.mem_get_info()[0]
with alloc.use_memory_pool(tag="weights"):
    g = torch.Generator(device="cuda"); g.manual_seed(7)
    w = torch.randint(1, 2**62, (GiB // 8,), device="cuda", dtype=torch.int64, generator=g)
with alloc.use_memory_pool(tag="kv"):
    kv = torch.zeros(2 * GiB // 2, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
w_sum = hashlib.sha256(w.cpu().numpy().tobytes()).hexdigest()
out["free_after_alloc_gib"] = round(torch.cuda.mem_get_info()[0] / GiB, 2)
w_ptr, kv_ptr = w.data_ptr(), kv.data_ptr()
alloc.sleep(offload_tags=("weights",)); torch.cuda.synchronize()
out["free_after_sleep_gib"] = round(torch.cuda.mem_get_info()[0] / GiB, 2)
out["freed_by_sleep_gib"] = round(out["free_after_sleep_gib"] - out["free_after_alloc_gib"], 2)
print("V0 sleep:", out, flush=True)

rc = lib.kfdr_checkpoint(1); out["kfdr_checkpoint_rc"] = rc
if rc: print("REPORT " + json.dumps(out), flush=True); os._exit(4)
r = lib.dmtcp_checkpoint()
if r == 1: print("source: checkpoint written, exiting", flush=True); os._exit(0)
t = time.time(); rr = lib.kfdr_restore(); out["kfdr_restore_rc"] = rr; out["kfdr_restore_s"] = round(time.time() - t, 3)
if rr: print("REPORT " + json.dumps(out), flush=True); os._exit(6)

checks = {}
t = time.time(); alloc.wake_up(); torch.cuda.synchronize(); out["wake_up_s"] = round(time.time() - t, 3)
checks["same_va"] = w.data_ptr() == w_ptr and kv.data_ptr() == kv_ptr
checks["weights_sha256"] = hashlib.sha256(w.cpu().numpy().tobytes()).hexdigest() == w_sum
kv.fill_(3.0); torch.cuda.synchronize()
# .item() on CuMemAllocator pool memory crashes on torch 2.13/ROCm even without C/R (CPU reads the VA): use .cpu().
checks["kv_rw"] = float(kv[123456:123457].cpu()[0]) == 3.0 and float(kv.float().sum().cpu()) == 3.0 * kv.numel()
y = (w[:1 << 20].float() * 0 + 1).sum(); checks["kernel_on_weights"] = float(y.cpu()) == float(1 << 20)
with alloc.use_memory_pool(tag="kv"):
    k2 = torch.full((GiB // 4,), 5, device="cuda", dtype=torch.int32)
checks["new_pool_alloc"] = int(k2[777:778].cpu()[0]) == 5
# second sleep/wake cycle, in the restored process
alloc.sleep(offload_tags=("weights",)); torch.cuda.synchronize(); alloc.wake_up(); torch.cuda.synchronize()
checks["second_sleep_wake"] = hashlib.sha256(w.cpu().numpy().tobytes()).hexdigest() == w_sum
try: lib.kfdr_drm_report()
except AttributeError: pass
out["checks"] = checks; out["pass"] = all(checks.values())
print("REPORT " + json.dumps(out), flush=True)
print("RESULT", "PASS" if out["pass"] else "FAIL", flush=True)
sys.exit(0 if out["pass"] else 1)
