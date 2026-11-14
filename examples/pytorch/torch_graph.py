# PyTorch process with a captured HIP graph survives a real cold restart (DMTCP + libkfdreplay).
#
#   scripts/cold_cycle.sh -d DIR -- python3 torch_graph.py DIR <seed>     (see run.sh)
#   dmtcp_restart ckpt_*.dmtcp
#
# Before the snapshot: model (bf16 MLP), warm-up, ONE graph capture, a known non-zero payload
# (PAYLOAD_GIB, pseudo-random, with checksum); weights are copied to the CPU for an independent reference.
# The source process exits (_exit) after dmtcp_checkpoint(). After restart:
#   - new deterministic inputs (seed from argv; they did not exist before the snapshot);
#   - outputs poisoned with NaN; graph.replay() without re-capture;
#   - comparison with an fp32 CPU reference, with bf16 tolerance;
#   - eager GEMM of a never-seen shape (hipBLASLt), new tensors/stream/event, a large allocation;
#   - payload checksum; recapture_count must stay 0;
#   - normal exit (destructors run).
# Result: a "REPORT {...}" JSON line and "RESULT PASS|FAIL". Chain mode: if <ckdir>/chain_left
# holds N > 0, the restored process checkpoints itself again (restore -> work -> checkpoint).
import ctypes, hashlib, json, os, sys, time

t_start = time.time()
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
import torch

ckdir, seed = sys.argv[1], int(sys.argv[2])
payload_gib = float(os.environ.get("PAYLOAD_GIB", "2"))
lib = ctypes.CDLL(None)
out = {"test_id": "pytorch-graph-cold", "seed": seed, "payload_gib": payload_gib, "alloc_conf": os.environ["PYTORCH_HIP_ALLOC_CONF"]}

# Capture counter: re-capturing after restore is not allowed.
captures = {"n": 0}
_cb = torch.cuda.CUDAGraph.capture_begin
def _counting_capture_begin(self, *a, **k):
    captures["n"] += 1
    return _cb(self, *a, **k)
torch.cuda.CUDAGraph.capture_begin = _counting_capture_begin

dev = torch.device("cuda")
torch.manual_seed(1234)
D, H, B = 2048, 8192, 256
model = torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.GELU(), torch.nn.Linear(H, D)).to(dev, torch.bfloat16).eval()
cpu_model = torch.nn.Sequential(torch.nn.Linear(D, H), torch.nn.GELU(), torch.nn.Linear(H, D))
cpu_model.load_state_dict({k: v.float().cpu() for k, v in model.state_dict().items()})

static_x = torch.randn(B, D, device=dev, dtype=torch.bfloat16)
static_y = torch.empty(B, D, device=dev, dtype=torch.bfloat16)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.no_grad(), torch.cuda.stream(s):
    for _ in range(3):
        static_y.copy_(model(static_x))
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.no_grad(), torch.cuda.graph(g):
    static_y.copy_(model(static_x))
g.replay(); torch.cuda.synchronize()

# Known non-zero payload: deterministic generator on the GPU, checksum computed on the CPU.
pay = None; pay_sum = None
if payload_gib > 0:
    n = int(payload_gib * (1 << 30)) // 8
    gen = torch.Generator(device=dev); gen.manual_seed(42)
    pay = torch.randint(1, 2**62, (n,), device=dev, dtype=torch.int64, generator=gen)
    torch.cuda.synchronize()
    pay_sum = hashlib.sha256(pay.cpu().numpy().tobytes()).hexdigest()
    out["payload_sha256"] = pay_sum
torch.cuda.synchronize()
out["captures_before"] = captures["n"]
out["warmup_s"] = round(time.time() - t_start, 3)
out["mem_allocated_mib"] = torch.cuda.memory_allocated() >> 20
out["mem_reserved_mib"] = torch.cuda.memory_reserved() >> 20

# --- cold checkpoint / restore; chain: restore -> work -> new checkpoint -> restore -----
def cold_checkpoint():
    t0 = time.time(); rc = lib.kfdr_checkpoint(1)
    out["kfdr_checkpoint_rc"] = rc; out["kfdr_checkpoint_s"] = round(time.time() - t0, 3)
    print("kfdr_checkpoint ->", rc, flush=True)
    if rc:
        print("REPORT " + json.dumps(out), flush=True); os._exit(4)
    t1 = time.time(); r = lib.dmtcp_checkpoint()
    if r == 1:
        print(f"source: checkpoint written in {time.time() - t1:.2f}s, exiting", flush=True)
        os._exit(0)
    out["t_abs_resumed"] = time.time()
    out["dmtcp_checkpoint_rc"] = r
    t0 = time.time(); rr = lib.kfdr_restore(); out["kfdr_restore_rc"] = rr; out["kfdr_restore_s"] = round(time.time() - t0, 3)
    print("kfdr_restore ->", rr, flush=True)
    if rr:
        print("REPORT " + json.dumps(out), flush=True); os._exit(6)

def run_checks(seed):
    checks = {}; t0 = time.time()
    gin = torch.Generator(); gin.manual_seed(seed)
    # 1) graph replay on new inputs, outputs poisoned.
    x_new = torch.randn(B, D, generator=gin).to(torch.bfloat16)
    static_x.copy_(x_new.to(dev)); static_y.fill_(float("nan"))
    g.replay(); torch.cuda.synchronize()
    y_graph = static_y.float().cpu()
    out["t_abs_graph_done"] = time.time()
    with torch.no_grad():
        y_ref = cpu_model(x_new.float())
    out["graph_median_relerr"] = ((y_graph - y_ref).abs() / (y_ref.abs() + 1e-2)).median().item()
    checks["graph_no_nan"] = bool(torch.isfinite(y_graph).all())
    checks["graph_vs_cpu_ref"] = bool(torch.allclose(y_graph, y_ref, rtol=5e-2, atol=5e-2))
    with torch.no_grad():
        y_eager = model(static_x).float().cpu()
    checks["graph_eq_eager_bitwise"] = bool(torch.equal(y_graph, y_eager))
    # 2) eager GEMM of a new shape (different per seed), new stream/event, large allocation.
    m = 3072 + 64 * (seed % 16)
    a = torch.randn(m, 1536, generator=gin); b2 = torch.randn(1536, 768, generator=gin)
    st = torch.cuda.Stream(); ev = torch.cuda.Event()
    with torch.cuda.stream(st):
        c = (a.to(dev, torch.bfloat16) @ b2.to(dev, torch.bfloat16)).float()
        ev.record(st)
    ev.synchronize()
    checks["new_gemm_vs_cpu"] = bool(torch.allclose(c.cpu(), a @ b2, rtol=5e-2, atol=1.0))
    big = torch.empty(1 << 30, device=dev, dtype=torch.uint8); big.fill_(7); torch.cuda.synchronize()
    checks["new_1gib_alloc"] = int(big[123456789].item()) == 7
    del big
    # 3) payload.
    if pay is not None:
        checks["payload_sha256"] = hashlib.sha256(pay.cpu().numpy().tobytes()).hexdigest() == pay_sum
    # 4) a second replay with other inputs.
    x2 = torch.randn(B, D, generator=gin).to(torch.bfloat16)
    static_x.copy_(x2.to(dev)); g.replay(); torch.cuda.synchronize()
    with torch.no_grad():
        checks["graph_second_inputs"] = bool(torch.allclose(static_y.float().cpu(), cpu_model(x2.float()), rtol=5e-2, atol=5e-2))
    out["post_restore_checks_s"] = round(time.time() - t0, 3)
    return checks

chain_file = os.path.join(ckdir, "chain_left")
generation = 0; all_ok = True
while True:
    cold_checkpoint()
    generation += 1
    checks = run_checks(seed + 1000 * generation)
    out["generation"] = generation; out["checks"] = checks
    out["captures_after"] = captures["n"]; out["recapture_count"] = captures["n"] - out["captures_before"]
    out["pid_after"] = os.getpid()
    try:
        lib.kfdr_drm_report()
    except AttributeError:
        pass
    ok = all(checks.values()) and out["recapture_count"] == 0
    all_ok = all_ok and ok
    out["pass"] = ok
    print("REPORT " + json.dumps(out), flush=True)
    left = int(open(chain_file).read()) if os.path.exists(chain_file) else 0
    if not ok or left <= 0:
        break
    open(chain_file, "w").write(str(left - 1))
    print(f"CHAIN generation {generation} ok, checkpointing again ({left - 1} left)", flush=True)
print("RESULT", "PASS" if all_ok else "FAIL", flush=True)
sys.exit(0 if all_ok else 1)
