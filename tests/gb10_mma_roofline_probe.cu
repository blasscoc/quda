#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void kernel_onchip_hmma(float *out, int iters)
{
  unsigned a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
  unsigned b0 = 0x3c003c00u, b1 = 0x3c003c00u;
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  for (int it = 0; it < iters; it++) {
#pragma unroll 16
    for (int u = 0; u < 16; u++) {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  if (c0 == -12345.0f) out[threadIdx.x] = c0 + c1 + c2 + c3;
}

// Streaming single-RHS Dslash-equivalent footprint:
// 65,536 sites per parity; each site reads ~576B (8 directions * (48B recon-8 link + 24B half-spinor)) + 48B spinor write = 624B/site
// Mode A: SIMT FFMA (576 FLOPs / site = 8 dirs * 72 FLOPs)
__global__ void kernel_stream_simt_dslash(const float4 *__restrict__ in_buf, float4 *__restrict__ out_buf, int n_sites)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_sites) return;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
  for (int d = 0; d < 8; d++) {
    float4 u0 = in_buf[d * n_sites + idx];
    float4 u1 = in_buf[((d + 1) & 7) * n_sites + idx];
    float4 v0 = in_buf[((d + 2) & 7) * n_sites + idx];
#pragma unroll
    for (int k = 0; k < 18; k++) {
      acc0 = fmaf(u0.x, v0.x, acc0);
      acc1 = fmaf(u0.y, v0.y, acc1);
      acc2 = fmaf(u1.z, v0.z, acc2);
      acc3 = fmaf(u1.w, v0.w, acc3);
    }
  }
  out_buf[idx] = make_float4(acc0, acc1, acc2, acc3);
}

// Mode B: Streaming single-RHS + warp-shuffle + 8 HMMA.16816.F32 per warp per direction (exact Mode 1 structure)
__global__ void kernel_stream_warp_hmma_dslash(const float4 *__restrict__ in_buf, float4 *__restrict__ out_buf, int n_sites)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_sites) return;
  int lane = threadIdx.x & 31;
  float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
#pragma unroll
  for (int d = 0; d < 8; d++) {
    float4 u0 = in_buf[d * n_sites + idx];
    float4 u1 = in_buf[((d + 1) & 7) * n_sites + idx];
    float4 v0 = in_buf[((d + 2) & 7) * n_sites + idx];
    __half2 h_u0 = __floats2half2_rn(u0.x, u0.y);
    __half2 h_u1 = __floats2half2_rn(u1.z, u1.w);
    __half2 h_v0 = __floats2half2_rn(v0.x, v0.y);
    unsigned ru0 = reinterpret_cast<unsigned &>(h_u0);
    unsigned ru1 = reinterpret_cast<unsigned &>(h_u1);
    unsigned rv0 = reinterpret_cast<unsigned &>(h_v0);
#pragma unroll
    for (int tile = 0; tile < 8; tile++) {
      unsigned a0 = __shfl_sync(0xffffffff, ru0, (tile * 4 + (lane & 3)) & 31);
      unsigned a1 = __shfl_sync(0xffffffff, ru1, (tile * 4 + ((lane + 1) & 3)) & 31);
      unsigned a2 = __shfl_sync(0xffffffff, ru0, (tile * 4 + ((lane + 2) & 3)) & 31);
      unsigned a3 = __shfl_sync(0xffffffff, ru1, (tile * 4 + ((lane + 3) & 3)) & 31);
      unsigned b0 = __shfl_sync(0xffffffff, rv0, (tile * 4 + (lane >> 3)) & 31);
      unsigned b1 = __shfl_sync(0xffffffff, rv0, (tile * 4 + ((lane >> 3) ^ 1)) & 31);
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(acc0), "+f"(acc1), "+f"(acc2), "+f"(acc3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      acc0 = __shfl_sync(0xffffffff, acc0, (lane + tile) & 31);
    }
  }
  out_buf[idx] = make_float4(acc0, acc1, acc2, acc3);
}

int main()
{
  const int n_sites = 65536; // 16^3 x 32 checkerboard sites
  float *d_dummy = nullptr;
  float4 *d_in = nullptr, *d_out = nullptr;
  cudaMalloc(&d_dummy, 1024 * sizeof(float));
  cudaMalloc(&d_in, 8 * n_sites * sizeof(float4));
  cudaMalloc(&d_out, n_sites * sizeof(float4));
  cudaMemset(d_in, 1, 8 * n_sites * sizeof(float4));

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  // 1. On-chip HMMA.16816.F32 peak probe
  const int blocks = 160, threads = 256, iters = 4096;
  kernel_onchip_hmma<<<blocks, threads>>>(d_dummy, 64);
  cudaDeviceSynchronize();
  cudaEventRecord(ev0);
  kernel_onchip_hmma<<<blocks, threads>>>(d_dummy, iters);
  cudaEventRecord(ev1);
  cudaEventSynchronize(ev1);
  float ms_hmma = 0.0f;
  cudaEventElapsedTime(&ms_hmma, ev0, ev1);
  double warps = (double)blocks * (threads / 32);
  double hmma_insts = warps * (double)iters * 16.0;
  double tflops_hmma = (hmma_insts * 4096.0) / (ms_hmma * 1e-3) / 1e12;

  // 2. Streaming SIMT vs Streaming Warp-Shuffle HMMA on 65,536 sites
  const int n_rep = 2000;
  kernel_stream_simt_dslash<<<n_sites / 256, 256>>>(d_in, d_out, n_sites);
  cudaDeviceSynchronize();
  cudaEventRecord(ev0);
  for (int r = 0; r < n_rep; r++) {
    kernel_stream_simt_dslash<<<n_sites / 256, 256>>>(d_in, d_out, n_sites);
  }
  cudaEventRecord(ev1);
  cudaEventSynchronize(ev1);
  float ms_simt = 0.0f;
  cudaEventElapsedTime(&ms_simt, ev0, ev1);

  kernel_stream_warp_hmma_dslash<<<n_sites / 256, 256>>>(d_in, d_out, n_sites);
  cudaDeviceSynchronize();
  cudaEventRecord(ev0);
  for (int r = 0; r < n_rep; r++) {
    kernel_stream_warp_hmma_dslash<<<n_sites / 256, 256>>>(d_in, d_out, n_sites);
  }
  cudaEventRecord(ev1);
  cudaEventSynchronize(ev1);
  float ms_tc_stream = 0.0f;
  cudaEventElapsedTime(&ms_tc_stream, ev0, ev1);

  double bytes_per_call = (double)(8 + 1) * n_sites * sizeof(float4);
  double us_simt = (ms_simt * 1000.0) / n_rep;
  double us_tc = (ms_tc_stream * 1000.0) / n_rep;
  double bw_simt_gbs = (bytes_per_call / (us_simt * 1e-6)) / 1e9;
  double bw_tc_gbs = (bytes_per_call / (us_tc * 1e-6)) / 1e9;

  std::printf("=== GB10 (sm_121) Tensor Core Roofline vs Streaming Dslash Micro-Probe ===\n");
  std::printf("ONCHIP_HMMA_16816_F32_TFLOPS=%.3f (elapsed_ms=%.3f, total_hmma=%.0f)\n", tflops_hmma, ms_hmma, hmma_insts);
  std::printf("STREAM_SIMT_DSLASH_US=%.3f (effective_bw_gbs=%.2f)\n", us_simt, bw_simt_gbs);
  std::printf("STREAM_WARP_HMMA_DSLASH_US=%.3f (effective_bw_gbs=%.2f, slowdown_vs_simt=%.3fx)\n",
              us_tc, bw_tc_gbs, us_tc / us_simt);
  std::printf("QUDA_ACTUAL_DSLASH_SIMT_US=52.85 (bw_gbs=345.0, from PRIOR_V2 profile)\n");
  std::printf("QUDA_ACTUAL_DSLASH_TC_MODE1_US=103.96 (bw_gbs=342.9, from work/tunecache/profile_0.tsv)\n");

  cudaFree(d_dummy);
  cudaFree(d_in);
  cudaFree(d_out);
  return 0;
}
