// HIP toy: libkfdreplay on non-idempotent kernels, AQL ring boundaries and scratch.
//
//   hipcc -O2 -o toy toy.hip
//   toy warm <minN> <residue>   drive the queue wptr to N >= minN with N mod L == residue,
//                               then checkpoint/restore inside the same process
//   toy cold <minN> <residue>   the same, cold: run under dmtcp_launch (scripts/cold_cycle.sh);
//                               the source process exits after the checkpoint, the checks run
//                               in the process started by dmtcp_restart
// residue < 0 means "L + residue" (e.g. -1 -> L-1). L is the ring size in packets.
#include <hip/hip_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

__global__ void tick(unsigned *counter, unsigned *marks, unsigned seq) {
  if (blockIdx.x == 0 && threadIdx.x == 0) { atomicAdd(counter, 1u); marks[seq % (1 << 20)] = seq; }
}
// Forces scratch use: a large private array indexed dynamically.
__global__ void scratchy(float *out, int k) {
  volatile float buf[2048];
  for (int i = 0; i < 2048; i++) buf[i] = (float)((i * 7 + k + threadIdx.x) % 1013);
  float s = 0; for (int i = 0; i < 2048; i += 1 + (k & 3)) s += buf[(i * 13 + threadIdx.x) % 2048];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
static float scratch_ref(int t, int k) {
  float buf[2048]; for (int i = 0; i < 2048; i++) buf[i] = (float)((i * 7 + k + t) % 1013);
  float s = 0; for (int i = 0; i < 2048; i += 1 + (k & 3)) s += buf[(i * 13 + t) % 2048]; return s;
}
#define CK(x) do { hipError_t e = (x); if (e) { printf("FAIL %s: %s\n", #x, hipGetErrorString(e)); return 3; } } while (0)

int main(int argc, char **argv) {
  bool cold = argc > 1 && !strcmp(argv[1], "cold");
  long minN = argc > 2 ? atol(argv[2]) : 0, residue = argc > 3 ? atol(argv[3]) : 1;
  auto ck = (int (*)(int))dlsym(RTLD_DEFAULT, "kfdr_checkpoint");
  auto rs = (int (*)())dlsym(RTLD_DEFAULT, "kfdr_restore");
  auto mw = (unsigned long (*)(unsigned *))dlsym(RTLD_DEFAULT, "kfdr_max_wptr");
  if (!ck || !rs || !mw) { printf("libkfdreplay is not preloaded\n"); return 2; }
  unsigned *counter, *marks; float *out; const int T = 256, B = 64;
  CK(hipMalloc(&counter, 4)); CK(hipMalloc(&marks, 4 << 20)); CK(hipMalloc(&out, T * B * 4));
  CK(hipMemset(counter, 0, 4)); CK(hipMemset(marks, 0xff, 4 << 20));
  scratchy<<<B, T>>>(out, 1); CK(hipDeviceSynchronize());
  unsigned seq = 0, L = 0;
  // Drive wptr to the requested ring position.
  for (;;) {
    unsigned long N = mw(&L);
    long want = residue < 0 ? (long)L + residue : residue;
    if ((long)N >= minN && L && (long)(N % L) == want) break;
    if ((long)N > minN + 4L * L) { printf("cannot hit residue, N=%lu L=%u\n", N, L); break; }
    tick<<<1, 1>>>(counter, marks, seq++);
    if ((seq & 255) == 0) CK(hipDeviceSynchronize());
  }
  CK(hipDeviceSynchronize());
  unsigned long N = mw(&L);
  printf("before C/R: launches %u, queue N=%lu L=%u N mod L=%lu\n", seq, N, L, L ? N % L : 0); fflush(stdout);
  int c = ck(cold ? 1 : 0); printf("checkpoint -> %d\n", c); fflush(stdout);
  if (c) return 4;
  if (cold) {
    auto dck = (int (*)())dlsym(RTLD_DEFAULT, "dmtcp_checkpoint");
    if (!dck) { printf("not under dmtcp_launch\n"); return 5; }
    int r = dck();  // 1: after the checkpoint (source process), 2: after restart
    printf("dmtcp_checkpoint -> %d (pid %d)\n", r, getpid()); fflush(stdout);
    if (r == 1) { printf("source process exits after checkpoint\n"); fflush(stdout); _exit(0); }
  }
  int r = rs(); printf("restore -> %d\n", r); fflush(stdout);
  if (r) return 6;
  // Non-idempotent check: exactly +K executions, markers in order, scratch kernel.
  const unsigned K = 1000;
  for (unsigned i = 0; i < K; i++) tick<<<1, 1>>>(counter, marks, seq++);
  scratchy<<<B, T>>>(out, 5);
  CK(hipDeviceSynchronize());
  unsigned got; CK(hipMemcpy(&got, counter, 4, hipMemcpyDeviceToHost));
  std::vector<unsigned> m(1 << 20); CK(hipMemcpy(m.data(), marks, 4 << 20, hipMemcpyDeviceToHost));
  std::vector<float> o(T * B); CK(hipMemcpy(o.data(), out, T * B * 4, hipMemcpyDeviceToHost));
  bool marks_ok = true; for (unsigned s = 0; s < seq; s++) if (m[s % (1 << 20)] != s) { marks_ok = false; break; }
  bool scr_ok = true; for (int i = 0; i < T * B; i++) if (o[i] != scratch_ref(i % T, 5)) { scr_ok = false; break; }
  unsigned long N2 = mw(&L);
  printf("after C/R: counter %u expected %u, marks %s, scratch %s, queue N=%lu\n", got, seq, marks_ok ? "ok" : "BAD",
         scr_ok ? "ok" : "BAD", N2);
  bool ok = got == seq && marks_ok && scr_ok;
  printf("RESULT %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
