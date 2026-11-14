# Cold restore of a vLLM engine (TP1, in-process V1) in sleep mode, through DMTCP + libkfdreplay.
#
#   python vllm_inprocess.py baseline <out.json>      plain process (enable_sleep_mode=True, no sleep): reference for set B
#   python vllm_inprocess.py sleepctl <baseline.json> control without C/R: warm-up -> export weights -> sleep(2) -> wake -> weights -> set B
#   (under dmtcp_launch, see run.sh) python vllm_inprocess.py cr <ckdir> <baseline.json>
#        warm-up (set A) -> copy model parameters and buffers to CPU -> sleep(level=2) ->
#        kfdr_checkpoint(1) -> dmtcp_checkpoint -> source exits; after dmtcp_restart: kfdr_restore -> wake_up(weights) ->
#        load weights from the CPU copy -> wake_up(kv_cache) -> set B, token-exact comparison with the reference.
# Not allowed after restore: loading weights from disk (DefaultModelLoader.load_weights / safe_open) and graph
# re-capture; both are counted in the report. Reloading from the in-process CPU copy is the warmctl flow (export -> reload).
import ctypes, json, os, sys, time

T0 = time.time()
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
mode = sys.argv[1]
MODEL = os.environ.get("KFDR_EX_MODEL", "Qwen/Qwen3-0.6B")
SLEEP_LEVEL = int(os.environ.get("KFDR_EX_SLEEP_LEVEL", "2"))

import torch
counters = {"capture_begin": 0, "load_weights": 0, "safe_open": 0}
_cb = torch.cuda.CUDAGraph.capture_begin
def _capture_begin(self, *a, **k):
    counters["capture_begin"] += 1
    return _cb(self, *a, **k)
torch.cuda.CUDAGraph.capture_begin = _capture_begin
import safetensors, safetensors.torch
_so = safetensors.safe_open
class _counting_safe_open:
    def __init__(self, *a, **k): counters["safe_open"] += 1; self._f = _so(*a, **k)
    def __enter__(self): return self._f.__enter__()
    def __exit__(self, *e): return self._f.__exit__(*e)
    def __getattr__(self, n): return getattr(self._f, n)
safetensors.safe_open = _counting_safe_open
safetensors.torch.safe_open = _counting_safe_open

from vllm import LLM, SamplingParams
from vllm.model_executor.model_loader import default_loader as _dl
_lw = _dl.DefaultModelLoader.load_weights
def _load_weights(self, *a, **k):
    counters["load_weights"] += 1
    return _lw(self, *a, **k)
_dl.DefaultModelLoader.load_weights = _load_weights

PROMPTS_A = [f"Write a short sentence about the number {i}." for i in range(8)]
PROMPTS_B = ["Explain in two sentences why the sky is blue.",
             "List three prime numbers greater than 100.",
             "Translate to French: The cat sleeps on the warm windowsill.",
             "What is 17 multiplied by 23? Answer with reasoning.",
             "Write a haiku about checkpointing a GPU process.",
             "Name the capital of Australia and one fact about it.",
             "Give a one-line Python function that reverses a string.",
             "Describe a sunrise over the ocean in one sentence.",
             "Why do computers use binary? Short answer.",
             "Summarize the plot of a heist movie in two sentences.",
             "What are the three states of matter?",
             "Suggest a name for a friendly robot and explain it."]
sp = SamplingParams(temperature=0.0, max_tokens=64, seed=0)
GiB = 1 << 30

def build():
    kw = dict(model=MODEL, max_model_len=2048, seed=0, enforce_eager=False, enable_sleep_mode=True)
    kvb = os.environ.get("KFDR_EX_KV_BYTES")
    if kvb: kw["kv_cache_memory_bytes"] = int(kvb)
    else: kw["gpu_memory_utilization"] = float(os.environ.get("KFDR_EX_GPU_UTIL", "0.05"))
    return LLM(**kw)

def gen(llm, prompts):
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in outs], [o.outputs[0].text for o in outs]

# Weight export/import run on the worker (in-process: the same process); the CPU copy lives in process memory
# and is part of the DMTCP image.
_CPU = {}
def _export(worker):
    m = worker.model_runner.model
    for n, t in list(m.named_parameters()) + [("buf." + n, b) for n, b in m.named_buffers()]:
        _CPU[n] = t.detach().to("cpu", copy=True)
    return sum(v.numel() * v.element_size() for v in _CPU.values())
def _import(worker):
    m = worker.model_runner.model; n_ok = 0
    with torch.no_grad():
        for n, t in list(m.named_parameters()) + [("buf." + n, b) for n, b in m.named_buffers()]:
            t.copy_(_CPU[n], non_blocking=True); n_ok += 1
    torch.cuda.synchronize()
    return n_ok

def free_gib(): return round(torch.cuda.mem_get_info()[0] / GiB, 2)

out = {"test_id": "vllm-inprocess-sleep-cold", "mode": mode, "model": MODEL, "sleep_level": SLEEP_LEVEL}
llm = build(); out["init_s"] = round(time.time() - T0, 2)
gen(llm, PROMPTS_A); torch.cuda.synchronize()
out["warm_ready_s"] = round(time.time() - T0, 2)
out["counters_before"] = dict(counters)
out["free_warm_gib"] = free_gib()

if mode == "baseline":
    t = time.time(); ids, texts = gen(llm, PROMPTS_B)
    out["first_B_generate_s"] = round(time.time() - t, 3)
    out["process_start_to_B_done_s"] = round(time.time() - T0, 2)
    out["ref_ids"] = ids; out["ref_texts"] = texts
    json.dump(out, open(sys.argv[2], "w"))
    print("BASELINE", json.dumps({k: v for k, v in out.items() if not k.startswith("ref_")}), flush=True)
    sys.exit(0)

def go_to_sleep():
    if SLEEP_LEVEL == 2:
        t = time.time(); out["export_bytes"] = llm.collective_rpc(_export)[0]; out["export_s"] = round(time.time() - t, 3)
    t = time.time(); llm.sleep(level=SLEEP_LEVEL); torch.cuda.synchronize(); out["sleep_s"] = round(time.time() - t, 3)
    out["free_asleep_gib"] = free_gib(); out["freed_by_sleep_gib"] = round(out["free_asleep_gib"] - out["free_warm_gib"], 2)
    out["torch_reserved_asleep_gib"] = round(torch.cuda.memory_reserved() / GiB, 2)
    print("asleep:", {k: out[k] for k in ("free_warm_gib", "free_asleep_gib", "freed_by_sleep_gib", "torch_reserved_asleep_gib")}, flush=True)

def wake():
    t = time.time()
    if SLEEP_LEVEL == 2:
        llm.wake_up(tags=["weights"]); out["import_tensors"] = llm.collective_rpc(_import)[0]
        llm.wake_up(tags=["kv_cache"])
    else:
        llm.wake_up()
    torch.cuda.synchronize(); out["wake_s"] = round(time.time() - t, 3)

def finish(ref):
    t = time.time(); ids, texts = gen(llm, PROMPTS_B); out["first_B_generate_s"] = round(time.time() - t, 3)
    out["t_abs_B_done"] = time.time()
    match = [a == b for a, b in zip(ids, ref["ref_ids"])]
    out["prompts_B_token_exact"] = f"{sum(match)}/{len(match)}"
    out["mismatch_examples"] = [{"got": texts[i][:120], "ref": ref["ref_texts"][i][:120]} for i, m in enumerate(match) if not m][:3]
    ids2, _ = gen(llm, PROMPTS_B); out["repeat_B_exact"] = ids2 == ref["ref_ids"]
    out["counters_after"] = dict(counters)
    out["reload_count"] = (counters["load_weights"] + counters["safe_open"]) - (out["counters_before"]["load_weights"] + out["counters_before"]["safe_open"])
    out["recapture_count"] = counters["capture_begin"] - out["counters_before"]["capture_begin"]
    ok = all(match) and out["repeat_B_exact"] and out["reload_count"] == 0 and out["recapture_count"] == 0
    out["pass"] = ok
    print("REPORT " + json.dumps(out), flush=True)
    print("RESULT", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)

if mode == "sleepctl":
    ref = json.load(open(sys.argv[2]))
    go_to_sleep(); wake(); finish(ref)

# --- cr ---------------------------------------------------------------------------
ckdir, ref = sys.argv[2], json.load(open(sys.argv[3]))
go_to_sleep()
lib = ctypes.CDLL(None)
t = time.time(); rc = lib.kfdr_checkpoint(1); out["kfdr_checkpoint_s"] = round(time.time() - t, 3); out["kfdr_checkpoint_rc"] = rc
print("kfdr_checkpoint ->", rc, flush=True)
if rc:
    print("REPORT " + json.dumps(out), flush=True); os._exit(4)
t = time.time(); r = lib.dmtcp_checkpoint()
if r == 1:
    print(f"source: checkpoint written in {time.time() - t:.2f}s, exiting", flush=True); os._exit(0)
out["t_abs_resumed"] = time.time()
t = time.time(); rr = lib.kfdr_restore(); out["kfdr_restore_s"] = round(time.time() - t, 3); out["kfdr_restore_rc"] = rr
print("kfdr_restore ->", rr, flush=True)
if rr:
    print("REPORT " + json.dumps(out), flush=True); os._exit(6)
wake()
try: lib.kfdr_drm_report()
except AttributeError: pass
finish(ref)
