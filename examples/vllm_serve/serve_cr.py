# `vllm serve` (OpenAI API, mp executor) through a cold checkpoint/restore on AMD, the warmctl flow:
# worker extension + /collective_rpc + /sleep + /wake_up, DMTCP snapshot of the process tree.
#
#   python serve_cr.py baseline <dir>   plain server (no DMTCP): warm-up A, answers to B -> <dir>/baseline.json
#   python serve_cr.py cr <dir> <baseline.json>
#        server under dmtcp_launch (LD_PRELOAD=libkfdreplay.so) -> warm-up A -> kfdr_arm -> kfdr_export -> /sleep?level=2
#        -> kfdr_checkpoint -> dmtcp_command --bcheckpoint -> --quit (source tree is dead) -> dmtcp_restart
#        -> kfdr_restore -> /wake_up?tags=weights -> kfdr_import -> /wake_up?tags=kv_cache -> B, compared with the reference.
# The orchestrator runs outside DMTCP; every HTTP request uses Connection: close (DMTCP cannot drain the socket
# of an external client). Paths: KFDR_LIB, DMTCP_ROOT, DMTCP_FDKEEP (see scripts/env.sh).
import glob, json, os, signal, subprocess, sys, time, urllib.request

MODE, DIR = sys.argv[1], sys.argv[2]
os.makedirs(DIR, exist_ok=True)
MODEL = os.environ.get("KFDR_EX_MODEL", "Qwen/Qwen3-0.6B")
W = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(W))
KFDR_LIB = os.environ.get("KFDR_LIB", os.path.join(REPO, "build", "libkfdreplay.so"))
D = os.path.join(os.environ.get("DMTCP_ROOT", "/opt/dmtcp"), "bin")
PLUGIN = os.environ.get("DMTCP_FDKEEP", "")
HTTP = int(os.environ.get("KFDR_EX_HTTP_PORT", "8000")); BASE = f"http://127.0.0.1:{HTTP}"
CPORT = 7000 + (os.getpid() % 2000)
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
out = {"test_id": "vllm-serve-sleep-cold", "mode": MODE, "model": MODEL}

def http(method, path, body=None, timeout=120):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Connection": "close"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        txt = r.read().decode()
    return json.loads(txt) if txt.strip().startswith(("{", "[")) else txt

def rpc(method):
    return http("POST", "/collective_rpc", {"method": method})["results"][0]

def wait_health(deadline_s):
    t = time.time()
    while time.time() - t < deadline_s:
        dead = [p for p in procs if p.poll() is not None]
        if dead: raise RuntimeError(f"server process exited rc={dead[0].returncode}")
        try:
            urllib.request.urlopen(urllib.request.Request(BASE + "/health", headers={"Connection": "close"}), timeout=5).read()
            return time.time() - t
        except Exception:
            time.sleep(0.05)
    raise TimeoutError("server not healthy")

def complete(prompts):
    r = http("POST", "/v1/completions", {"model": MODEL, "prompt": prompts, "temperature": 0.0, "max_tokens": 64, "seed": 0})
    return [c["text"] for c in sorted(r["choices"], key=lambda c: c["index"])]

def serve_cmd():
    return ["vllm", "serve", MODEL, "--port", str(HTTP), "--max-model-len", "2048", "--seed", "0",
            "--gpu-memory-utilization", os.environ.get("KFDR_EX_GPU_UTIL", "0.05"), "--enable-sleep-mode",
            "--worker-extension-cls", "kfdr_ext.KfdrExt"]

env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", PYTHONPATH=W + ":" + os.environ.get("PYTHONPATH", ""),
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", UV_USE_IO_URING="0",
           UCX_MEM_MMAP_HOOK_MODE="none", USE_LIBUV="0", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
           HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "0" if MODE == "baseline" else "1"),
           HSA_ENABLE_SDMA="0", HSA_ENABLE_SCRATCH_ASYNC_RECLAIM="0", HSA_NO_SCRATCH_RECLAIM="1")
procs = []
def spawn(cmd, logname, extra_env=None):
    p = subprocess.Popen(cmd, stdout=open(os.path.join(DIR, logname), "w"), stderr=subprocess.STDOUT,
                         env=dict(env, **(extra_env or {})), start_new_session=True)
    procs.append(p); return p
def cleanup():
    for p in procs:
        try: os.killpg(p.pid, signal.SIGKILL)
        except Exception: pass
    subprocess.run(["pkill", "-9", "-f", "VLLM::EngineCor"]); subprocess.run(["pkill", "-9", "-f", f"dmtcp_coordinator --port {CPORT}"])
    subprocess.run(["pkill", "-9", "-f", f"dmtcp_coordinator --port {CPORT + 1}"])
def finish(code):
    print("REPORT " + json.dumps(out), flush=True)
    print("RESULT", "PASS" if out.get("pass") else "FAIL", flush=True)
    cleanup(); sys.exit(code)

try:
    T0 = time.time()
    if MODE == "baseline":
        spawn(serve_cmd(), "server.txt")
        out["start_to_healthy_s"] = round(wait_health(600), 2)
        complete(PROMPTS_A)
        t = time.time(); out["ref_texts"] = complete(PROMPTS_B); out["B_s"] = round(time.time() - t, 3)
        out["start_to_B_done_s"] = round(time.time() - T0, 2); out["pass"] = True
        json.dump(out, open(os.path.join(DIR, "baseline.json"), "w")); finish(0)

    ref = json.load(open(sys.argv[3]))["ref_texts"]
    spawn([f"{D}/dmtcp_launch", "--no-gzip"] + (["--with-plugin", PLUGIN] if PLUGIN else []) + ["--new-coordinator", "-p", str(CPORT),
           "--ckptdir", DIR] + serve_cmd(), "src.txt", {"LD_PRELOAD": KFDR_LIB})
    out["start_to_healthy_s"] = round(wait_health(600), 2)
    complete(PROMPTS_A)
    rpc("kfdr_arm"); out["engine_pid"] = rpc("kfdr_pid")
    out["free_warm"] = rpc("kfdr_free")
    out["export"] = rpc("kfdr_export")
    t = time.time(); http("POST", "/sleep?level=2"); out["sleep_s"] = round(time.time() - t, 3)
    out["free_asleep"] = rpc("kfdr_free")
    out["kfdr_checkpoint"] = rpc("kfdr_checkpoint")
    if out["kfdr_checkpoint"]["rc"]: finish(4)
    t = time.time()
    r = subprocess.run([f"{D}/dmtcp_command", "-p", str(CPORT), "--bcheckpoint"], capture_output=True, text=True, timeout=600)
    out["dmtcp_checkpoint_s"] = round(time.time() - t, 2); out["dmtcp_checkpoint_rc"] = r.returncode
    subprocess.run([f"{D}/dmtcp_command", "-p", str(CPORT), "--quit"], capture_output=True, timeout=60)
    time.sleep(1); cleanup(); procs.clear()
    imgs = sorted(glob.glob(os.path.join(DIR, "ckpt_*.dmtcp")))
    out["images"] = len(imgs); out["image_mib"] = sum(os.path.getsize(f) for f in imgs) >> 20
    if not imgs: finish(5)
    if os.environ.get("EVICT"):
        for f in imgs:
            fd = os.open(f, os.O_RDONLY); os.fsync(fd); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        out["evicted"] = True
    alive = subprocess.run(["pgrep", "-f", "VLLM::EngineCor|vllm serve"], capture_output=True, text=True).stdout.split()
    out["alive_before_restart"] = len(alive)
    t2 = time.time()
    spawn([f"{D}/dmtcp_restart", "--no-strict-checking", "--new-coordinator", "-p", str(CPORT + 1)] + imgs, "dst.txt",
          {"DMTCP_RESTART_HUGEPAGES": "1", "DMTCP_RESTART_READERS": "16", "DMTCP_RESTART_MMAP": "0"})
    out["restart_to_healthy_s"] = round(wait_health(300), 3)
    out["kfdr_restore"] = rpc("kfdr_restore")
    if out["kfdr_restore"]["rc"]: finish(6)
    t = time.time(); http("POST", "/wake_up?tags=weights"); out["import"] = rpc("kfdr_import")
    http("POST", "/wake_up?tags=kv_cache"); out["wake_s"] = round(time.time() - t, 3)
    t = time.time(); got = complete(PROMPTS_B); out["B_s"] = round(time.time() - t, 3)
    out["restart_to_B_done_s"] = round(time.time() - t2, 3)
    match = [a == b for a, b in zip(got, ref)]
    out["prompts_B_text_exact"] = f"{sum(match)}/{len(match)}"
    out["mismatch_examples"] = [{"got": got[i][:120], "ref": ref[i][:120]} for i, m in enumerate(match) if not m][:3]
    out["repeat_B_exact"] = complete(PROMPTS_B) == ref
    out["counters"] = rpc("kfdr_counters"); out["engine_pid_after"] = rpc("kfdr_pid")
    out["pass"] = all(match) and out["repeat_B_exact"] and out["counters"].get("load_weights") == 0 and out["counters"].get("capture_begin") == 0
    if not os.environ.get("KEEP_CKPT") and out["pass"]:
        for f in imgs: os.remove(f)
    finish(0 if out["pass"] else 1)
except Exception as e:
    out["error"] = repr(e); finish(2)
