# Like vllm_inprocess.py (sleep(2) -> cold C/R -> wake -> reload weights), but with the mp executor:
# VLLM_ENABLE_V1_MULTIPROCESSING=1, EngineCore (and the GPU) in a child process, the LLM frontend in the parent.
#
#   python vllm_mp.py baseline <out.json>
#   python vllm_mp.py sleepctl <baseline.json>
#   (under dmtcp_launch; LD_PRELOAD is inherited by the child) python vllm_mp.py cr <ckdir> <baseline.json>
#
# kfdr_checkpoint/kfdr_restore run in the EngineCore process via collective_rpc; dmtcp_checkpoint is called
# by the parent and DMTCP snapshots the whole tree. The parent never touches the GPU (no torch.cuda calls).
# Counters (weight loads, graph captures) are installed in EngineCore after warm-up and must stay zero.
import ctypes, json, os, sys, time

GiB = 1 << 30
# EngineCore-side state lives on the worker object and in the torch module: cloudpickle ships functions from
# __main__ by value together with their globals, so module-level dicts would be a fresh copy on every RPC.

# --- functions executed on the worker (in the EngineCore process) --------------------------------
def w_pid(worker): return os.getpid()
def w_install_counters(worker):
    import torch
    from vllm.model_executor.model_loader import default_loader as dl
    if hasattr(torch.cuda.CUDAGraph, "_b3c_cnt"): return True
    cnt = {"capture_begin": 0, "load_weights": 0}; torch.cuda.CUDAGraph._b3c_cnt = cnt
    cb, lw = torch.cuda.CUDAGraph.capture_begin, dl.DefaultModelLoader.load_weights
    def capture_begin(self, *a, **k): cnt["capture_begin"] += 1; return cb(self, *a, **k)
    def load_weights(self, *a, **k): cnt["load_weights"] += 1; return lw(self, *a, **k)
    torch.cuda.CUDAGraph.capture_begin = capture_begin
    dl.DefaultModelLoader.load_weights = load_weights
    return True
def w_counters(worker):
    import torch; return dict(torch.cuda.CUDAGraph._b3c_cnt)
def w_free_gib(worker):
    import torch; torch.cuda.synchronize()
    return round(torch.cuda.mem_get_info()[0] / GiB, 2), round(torch.cuda.memory_reserved() / GiB, 2)
def w_export(worker):
    m = worker.model_runner.model; cpu = worker._b3c_cpu = {}
    for n, t in list(m.named_parameters()) + [("buf." + n, b) for n, b in m.named_buffers()]:
        cpu[n] = t.detach().to("cpu", copy=True)
    return sum(v.numel() * v.element_size() for v in cpu.values())
def w_import(worker):
    import torch
    m = worker.model_runner.model; n_ok = 0
    with torch.no_grad():
        for n, t in list(m.named_parameters()) + [("buf." + n, b) for n, b in m.named_buffers()]:
            t.copy_(worker._b3c_cpu[n], non_blocking=True); n_ok += 1
    torch.cuda.synchronize()
    return n_ok
def w_kfdr_checkpoint(worker):
    import torch; torch.cuda.synchronize()
    t = time.time(); rc = ctypes.CDLL(None).kfdr_checkpoint(1); return rc, round(time.time() - t, 3)
def w_kfdr_restore(worker):
    t = time.time(); rc = ctypes.CDLL(None).kfdr_restore(); return rc, round(time.time() - t, 3)
def w_drm_report(worker):
    try: ctypes.CDLL(None).kfdr_drm_report()
    except AttributeError: pass
    return True

def main():
    T0 = time.time()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    # collective_rpc with a function across processes needs pickle serialization in vLLM 0.29 (test only;
    # a real integration uses worker-extension methods, see examples/vllm_serve/kfdr_ext.py).
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    mode = sys.argv[1]
    MODEL = os.environ.get("KFDR_EX_MODEL", "Qwen/Qwen3-0.6B")
    from vllm import LLM, SamplingParams
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
    kw = dict(model=MODEL, max_model_len=2048, seed=0, enforce_eager=False, enable_sleep_mode=True,
              gpu_memory_utilization=float(os.environ.get("KFDR_EX_GPU_UTIL", "0.05")))
    llm = LLM(**kw)
    rpc = lambda f: llm.collective_rpc(f)[0]
    def die(code):
        # os._exit does not stop the EngineCore child: it would stay orphaned and keep the GPU.
        try: os.kill(out["engine_pid"], 9)
        except Exception: pass
        os._exit(code)
    def gen(prompts):
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs], [o.outputs[0].text for o in outs]
    out = {"test_id": "vllm-mp-sleep-cold", "mode": mode, "model": MODEL, "main_pid": os.getpid()}
    out["init_s"] = round(time.time() - T0, 2)
    out["engine_pid"] = rpc(w_pid)
    gen(PROMPTS_A)
    out["warm_ready_s"] = round(time.time() - T0, 2)
    rpc(w_install_counters)
    out["free_warm_gib"], _ = rpc(w_free_gib)

    if mode == "baseline":
        t = time.time(); ids, texts = gen(PROMPTS_B)
        out["first_B_generate_s"] = round(time.time() - t, 3); out["process_start_to_B_done_s"] = round(time.time() - T0, 2)
        out["ref_ids"] = ids; out["ref_texts"] = texts
        json.dump(out, open(sys.argv[2], "w"))
        print("BASELINE", json.dumps({k: v for k, v in out.items() if not k.startswith("ref_")}), flush=True)
        die(0)

    def go_to_sleep():
        t = time.time(); out["export_bytes"] = rpc(w_export); out["export_s"] = round(time.time() - t, 3)
        t = time.time(); llm.sleep(level=2); out["sleep_s"] = round(time.time() - t, 3)
        out["free_asleep_gib"], out["torch_reserved_asleep_gib"] = rpc(w_free_gib)
        out["freed_by_sleep_gib"] = round(out["free_asleep_gib"] - out["free_warm_gib"], 2)
        print("asleep:", {k: out[k] for k in ("free_warm_gib", "free_asleep_gib", "freed_by_sleep_gib", "torch_reserved_asleep_gib")}, flush=True)
    def wake():
        t = time.time(); llm.wake_up(tags=["weights"]); out["import_tensors"] = rpc(w_import)
        llm.wake_up(tags=["kv_cache"]); out["wake_s"] = round(time.time() - t, 3)
    def finish(ref):
        t = time.time(); ids, texts = gen(PROMPTS_B); out["first_B_generate_s"] = round(time.time() - t, 3)
        out["t_abs_B_done"] = time.time()
        match = [a == b for a, b in zip(ids, ref["ref_ids"])]
        out["prompts_B_token_exact"] = f"{sum(match)}/{len(match)}"
        out["mismatch_examples"] = [{"got": texts[i][:120], "ref": ref["ref_texts"][i][:120]} for i, m in enumerate(match) if not m][:3]
        ids2, _ = gen(PROMPTS_B); out["repeat_B_exact"] = ids2 == ref["ref_ids"]
        c = rpc(w_counters); out["counters_after"] = c
        out["reload_count"] = c["load_weights"]; out["recapture_count"] = c["capture_begin"]
        out["engine_pid_after"] = rpc(w_pid)
        ok = all(match) and out["repeat_B_exact"] and out["reload_count"] == 0 and out["recapture_count"] == 0
        out["pass"] = ok
        print("REPORT " + json.dumps(out), flush=True)
        print("RESULT", "PASS" if ok else "FAIL", flush=True)
        die(0 if ok else 1)

    if mode == "sleepctl":
        ref = json.load(open(sys.argv[2])); go_to_sleep(); wake(); finish(ref)

    # --- cr --------------------------------------------------------------------------
    ckdir, ref = sys.argv[2], json.load(open(sys.argv[3]))
    go_to_sleep()
    rc, out["kfdr_checkpoint_s"] = rpc(w_kfdr_checkpoint); out["kfdr_checkpoint_rc"] = rc
    print("engine kfdr_checkpoint ->", rc, flush=True)
    if rc:
        print("REPORT " + json.dumps(out), flush=True); die(4)
    lib = ctypes.CDLL(None)
    t = time.time(); r = lib.dmtcp_checkpoint()
    if r == 1:
        print(f"source: checkpoint written in {time.time() - t:.2f}s, killing engine {out['engine_pid']} and exiting", flush=True)
        try: os.kill(out["engine_pid"], 9)
        except OSError as e: print("kill engine:", e, flush=True)
        os._exit(0)
    out["t_abs_resumed"] = time.time()
    rr, out["kfdr_restore_s"] = rpc(w_kfdr_restore); out["kfdr_restore_rc"] = rr
    print("engine kfdr_restore ->", rr, flush=True)
    if rr:
        print("REPORT " + json.dumps(out), flush=True); die(6)
    wake(); rpc(w_drm_report); finish(ref)

if __name__ == "__main__":
    main()
