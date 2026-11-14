"""vLLM worker extension for cold checkpoint/restore on AMD (libkfdreplay + DMTCP), the warmctl scheme:
`vllm serve ... --worker-extension-cls kfdr_ext.KfdrExt`; the host calls the methods with POST /collective_rpc
(requires VLLM_SERVER_DEV_MODE=1). Methods run in the worker process (EngineCore with TP1).

  kfdr_export     copy model parameters and buffers to CPU (process memory, part of the DMTCP image)
  kfdr_import     copy them back (after wake_up(tags=["weights"]))
  kfdr_checkpoint freeze and dump the GPU state of the process (cold), before `dmtcp_command --bcheckpoint`
  kfdr_restore    re-create the KFD objects in the new process (right after dmtcp_restart)
  kfdr_counters   number of weight loads from disk and graph captures since kfdr_arm
  kfdr_arm        install the counters (after warm-up)
"""
import ctypes, os, time


class KfdrExt:
    def kfdr_pid(self):
        return os.getpid()

    def kfdr_arm(self):
        import torch
        from vllm.model_executor.model_loader import default_loader as dl
        if hasattr(torch.cuda.CUDAGraph, "_kfdr_cnt"):
            return True
        cnt = {"capture_begin": 0, "load_weights": 0}
        torch.cuda.CUDAGraph._kfdr_cnt = cnt
        cb, lw = torch.cuda.CUDAGraph.capture_begin, dl.DefaultModelLoader.load_weights
        def capture_begin(g, *a, **k):
            cnt["capture_begin"] += 1
            return cb(g, *a, **k)
        def load_weights(l, *a, **k):
            cnt["load_weights"] += 1
            return lw(l, *a, **k)
        torch.cuda.CUDAGraph.capture_begin = capture_begin
        dl.DefaultModelLoader.load_weights = load_weights
        return True

    def kfdr_counters(self):
        import torch
        return dict(getattr(torch.cuda.CUDAGraph, "_kfdr_cnt", {}))

    def _kfdr_tensors(self):
        m = self.model_runner.model
        return list(m.named_parameters()) + [("buf." + n, b) for n, b in m.named_buffers()]

    def kfdr_export(self):
        t = time.time()
        self._kfdr_cpu = {n: x.detach().to("cpu", copy=True) for n, x in self._kfdr_tensors()}
        return {"bytes": sum(v.numel() * v.element_size() for v in self._kfdr_cpu.values()), "s": round(time.time() - t, 3)}

    def kfdr_import(self):
        import torch
        t = time.time()
        with torch.no_grad():
            for n, x in self._kfdr_tensors():
                x.copy_(self._kfdr_cpu[n], non_blocking=True)
        torch.cuda.synchronize()
        return {"tensors": len(self._kfdr_cpu), "s": round(time.time() - t, 3)}

    def kfdr_checkpoint(self):
        import torch
        torch.cuda.synchronize()
        t = time.time()
        rc = ctypes.CDLL(None).kfdr_checkpoint(1)
        return {"rc": rc, "s": round(time.time() - t, 3)}

    def kfdr_restore(self):
        t = time.time()
        rc = ctypes.CDLL(None).kfdr_restore()
        return {"rc": rc, "s": round(time.time() - t, 3)}

    def kfdr_free(self):
        import torch
        torch.cuda.synchronize()
        return {"free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
                "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2)}
