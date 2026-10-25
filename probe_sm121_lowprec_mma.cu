#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

struct ComplexF {
  float re, im;
};

struct SU3Link {
  ComplexF e[3][3];
};

struct HalfSpinor {
  ComplexF c[2][3]; // 2 spins, 3 colors
};

__device__ __forceinline__ unsigned shfl_u32(unsigned v, int src) {
  unsigned r;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(r) : "r"(v), "r"(src));
  return r;
}

__device__ __forceinline__ float shfl_f32(float v, int src) {
  float r;
  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(r) : "f"(v), "r"(src));
  return r;
}

__device__ __forceinline__ unsigned pack_f16x2(float re, float im) {
  __half2 h = __floats2half2_rn(re, im);
  return reinterpret_cast<unsigned &>(h);
}

__device__ __forceinline__ unsigned format_b_16x2(unsigned pk, int is_im_col) {
  return is_im_col ? ((pk >> 16) | (pk << 16)) : (pk ^ 0x80000000u);
}

__device__ __forceinline__ unsigned pack_bf16x2(float re, float im) {
  __nv_bfloat162 b = __floats2bfloat162_rn(re, im);
  return reinterpret_cast<unsigned &>(b);
}

__device__ __forceinline__ unsigned pack_e4m3x2_in_u32(float x0, float x1) {
  unsigned short s;
  asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n" : "=h"(s) : "f"(x0), "f"(x1));
  return (unsigned)s; // bytes 0,1 populated, bytes 2,3 = 0
}

__device__ __forceinline__ unsigned format_b_fp8x2(unsigned pk16, int is_im_col) {
  unsigned b0 = pk16 & 0xffu;
  unsigned b1 = (pk16 >> 8) & 0xffu;
  return is_im_col ? (b1 | (b0 << 8)) : (b0 | ((b1 ^ 0x80u) << 8));
}

__device__ __forceinline__ unsigned pack_e5m2x2_in_u32(float x0, float x1) {
  unsigned short s;
  asm("{cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;}\n" : "=h"(s) : "f"(x0), "f"(x1));
  return (unsigned)s;
}

__device__ __forceinline__ unsigned pack_e2m1x2_in_u32(float x0, float x1) {
  unsigned short s;
  asm("{ .reg .b8 __t; cvt.rn.satfinite.e2m1x2.f32 __t, %2, %1; mov.b16 %0, {__t, 0}; }\n"
      : "=h"(s) : "f"(x0), "f"(x1));
  return (unsigned)(s & 0xffu); // low byte holds 2 nibbles: [3:0]=x0, [7:4]=x1; bytes 1..3 = 0
}

__device__ __forceinline__ unsigned format_b_fp4x2(unsigned pk8, int is_im_col) {
  unsigned n0 = pk8 & 0xfu;
  unsigned n1 = (pk8 >> 4) & 0xfu;
  return is_im_col ? (n1 | (n0 << 4)) : (n0 | ((n1 ^ 0x8u) << 4));
}

// Generic 4-site-per-instruction warp contraction template for m16n8k16 / m16n8k32 / m16n8k64
// Mode:
// 0: FP32 SIMT
// 1: FP16 HMMA.16816.F32
// 2: BF16 HMMA.16816.F32.BF16
// 3: TF32 HMMA.1688.F32.TF32
// 4: FP8 E4M3 QMMA.16832.F32 (with per-site exponent scale on spinor & 2.0x scale on U)
// 5: FP8 E5M2 QMMA.16832.F32
// 6: MXFP8 E4M3 QMMA.SF.16832.F32.E4M3.E4M3.E8 (block_scale ue8m0)
// 7: NVFP4 E2M1 OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X (block_scale ue4m3, scaled by 6.0)
// 8: INT8 Quarter fixed-point (round(127*x)) sloppy contraction control
template <int MODE>
__device__ __forceinline__ HalfSpinor contract_su3_halfspinor(const SU3Link &U, const HalfSpinor &in) {
  HalfSpinor out;
  if constexpr (MODE == 0) {
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        float re = 0.0f, im = 0.0f;
        #pragma unroll
        for (int j = 0; j < 3; j++) {
          re += U.e[i][j].re * in.c[s][j].re - U.e[i][j].im * in.c[s][j].im;
          im += U.e[i][j].re * in.c[s][j].im + U.e[i][j].im * in.c[s][j].re;
        }
        out.c[s][i] = {re, im};
      }
    }
    return out;
  } else if constexpr (MODE == 8) {
    // 8-bit Quarter Fixed-Point (int8_t, round(127 * x)) for both U (in [-1,1]) and normalized spinor
    float vmax = 0.0f;
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) {
        vmax = fmaxf(vmax, fabsf(in.c[s][j].re));
        vmax = fmaxf(vmax, fabsf(in.c[s][j].im));
      }
    }
    float v_scale = (vmax > 0.0f) ? (127.0f / vmax) : 0.0f;
    float unscale = (vmax > 0.0f) ? (vmax / (127.0f * 127.0f)) : 0.0f;
    int8_t u_q[3][3][2];
    int8_t v_q[2][3][2];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) {
        u_q[i][j][0] = (int8_t)rintf(fminf(fmaxf(U.e[i][j].re * 127.0f, -127.0f), 127.0f));
        u_q[i][j][1] = (int8_t)rintf(fminf(fmaxf(U.e[i][j].im * 127.0f, -127.0f), 127.0f));
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) {
        v_q[s][j][0] = (int8_t)rintf(fminf(fmaxf(in.c[s][j].re * v_scale, -127.0f), 127.0f));
        v_q[s][j][1] = (int8_t)rintf(fminf(fmaxf(in.c[s][j].im * v_scale, -127.0f), 127.0f));
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        int acc_re = 0, acc_im = 0;
        #pragma unroll
        for (int j = 0; j < 3; j++) {
          acc_re += (int)u_q[i][j][0] * (int)v_q[s][j][0] - (int)u_q[i][j][1] * (int)v_q[s][j][1];
          acc_im += (int)u_q[i][j][0] * (int)v_q[s][j][1] + (int)u_q[i][j][1] * (int)v_q[s][j][0];
        }
        out.c[s][i] = {(float)acc_re * unscale, (float)acc_im * unscale};
      }
    }
    return out;
  } else if constexpr (MODE == 3) {
    // TF32 m16n8k8 (2 sites per instruction)
    int lane_id = threadIdx.x & 31;
    int group_id = lane_id >> 2;
    int tid = lane_id & 3;
    int g_low = group_id & 3;
    int g_high = group_id >> 2;
    unsigned u_tf[3][6], v_tf[2][6];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) {
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u_tf[i][2*j+0]) : "f"(U.e[i][j].re));
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(u_tf[i][2*j+1]) : "f"(U.e[i][j].im));
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) {
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(v_tf[s][2*j+0]) : "f"(in.c[s][j].re));
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(v_tf[s][2*j+1]) : "f"(in.c[s][j].im));
      }
    }
    int spin_col = (g_low >> 1) & 1;
    int is_im_col = g_low & 1;
    int pair_id = lane_id >> 1;
    int q_pair = lane_id & 1;
    float out_re[6] = {0}, out_im[6] = {0};
    #pragma unroll
    for (int tile = 0; tile < 16; tile++) {
      int s0 = 2 * tile, s1 = 2 * tile + 1;
      unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int k = 0; k < 6; k++) {
          unsigned u0 = shfl_u32(u_tf[i][k], s0);
          unsigned u1 = shfl_u32(u_tf[i][k], s1);
          bool m_i = (group_id == i);
          if (m_i && tid == k) { a0 = u0; a1 = u1; }
          if (m_i && tid + 4 == k) { a2 = u0; a3 = u1; }
        }
      }
      int sb = (g_high == 0) ? s0 : s1;
      unsigned b0 = 0, b1 = 0;
      #pragma unroll
      for (int s = 0; s < 2; s++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
          unsigned vr = shfl_u32(v_tf[s][2*j+0], sb);
          unsigned vi = shfl_u32(v_tf[s][2*j+1], sb);
          unsigned ke = is_im_col ? vi : vr;
          unsigned ko = is_im_col ? vr : (vi ^ 0x80000000u);
          if (spin_col == s && tid == 2*j+0) b0 = ke;
          if (spin_col == s && tid == 2*j+1) b0 = ko;
          if (spin_col == s && tid + 4 == 2*j+0) b1 = ke;
          if (spin_col == s && tid + 4 == 2*j+1) b1 = ko;
        }
      }
      float c[4] = {0, 0, 0, 0};
      asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      float l_re = (tid < 2) ? c[0] : c[2];
      float l_im = (tid < 2) ? c[1] : c[3];
      #pragma unroll
      for (int s = 0; s < 2; s++) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          int sl = 4 * i + 2 * q_pair + s;
          float rr = shfl_f32(l_re, sl);
          float ri = shfl_f32(l_im, sl);
          if (pair_id == tile) {
            out_re[s*3+i] = rr;
            out_im[s*3+i] = ri;
          }
        }
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; s++)
      #pragma unroll
      for (int i = 0; i < 3; i++)
        out.c[s][i] = {out_re[s*3+i], out_im[s*3+i]};
    return out;
  } else {
    // 4-sites-per-instruction warp-collaborative layout for FP16 (1), BF16 (2), FP8 E4M3 (4), FP8 E5M2 (5), MXFP8 (6), NVFP4 (7)
    int lane_id = threadIdx.x & 31;
    int group_id = lane_id >> 2;
    int tid = lane_id & 3;
    int g_low = group_id & 3;
    int g_high = group_id >> 2;

    float vmax = 0.0f;
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) {
        vmax = fmaxf(vmax, fabsf(in.c[s][j].re));
        vmax = fmaxf(vmax, fabsf(in.c[s][j].im));
      }
    }
    // For NVFP4 (e2m1, maxnorm=6.0), scale inputs into [-6, 6] so we use all dynamic levels {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    constexpr float target_max = (MODE == 7) ? 6.0f : 1.0f;
    constexpr float u_prescale = (MODE == 7) ? 6.0f : 1.0f;
    constexpr float inv_u_prescale = 1.0f / u_prescale;

    float inv_vmax = (vmax > 0.0f) ? (target_max / vmax) : 0.0f;
    float unscale = (vmax > 0.0f) ? ((vmax / target_max) * inv_u_prescale) : 0.0f;

    unsigned u_pk[9], v_pk[6];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) {
        float ur = U.e[i][j].re * u_prescale;
        float ui = U.e[i][j].im * u_prescale;
        if constexpr (MODE == 1) u_pk[3*i+j] = pack_f16x2(ur, ui);
        else if constexpr (MODE == 2) u_pk[3*i+j] = pack_bf16x2(ur, ui);
        else if constexpr (MODE == 4 || MODE == 6) u_pk[3*i+j] = pack_e4m3x2_in_u32(ur, ui);
        else if constexpr (MODE == 5) u_pk[3*i+j] = pack_e5m2x2_in_u32(ur, ui);
        else if constexpr (MODE == 7) u_pk[3*i+j] = pack_e2m1x2_in_u32(ur, ui);
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; s++) {
      #pragma unroll
      for (int j = 0; j < 3; j++) {
        float vr = in.c[s][j].re * inv_vmax;
        float vi = in.c[s][j].im * inv_vmax;
        if constexpr (MODE == 1) v_pk[3*s+j] = pack_f16x2(vr, vi);
        else if constexpr (MODE == 2) v_pk[3*s+j] = pack_bf16x2(vr, vi);
        else if constexpr (MODE == 4 || MODE == 6) v_pk[3*s+j] = pack_e4m3x2_in_u32(vr, vi);
        else if constexpr (MODE == 5) v_pk[3*s+j] = pack_e5m2x2_in_u32(vr, vi);
        else if constexpr (MODE == 7) v_pk[3*s+j] = pack_e2m1x2_in_u32(vr, vi);
      }
    }

    int spin_col = (g_low >> 1) & 1;
    int is_im_col = g_low & 1;
    float out_re[6] = {0}, out_im[6] = {0};

    #pragma unroll
    for (int tile = 0; tile < 8; tile++) {
      int src_a_even = 4 * tile + g_high;
      int src_a_odd  = 4 * tile + g_high + 2;
      unsigned a_even = 0u, a_odd = 0u;
      #pragma unroll
      for (int i = 0; i < 3; i++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
          unsigned ue = shfl_u32(u_pk[3*i+j], src_a_even);
          unsigned uo = shfl_u32(u_pk[3*i+j], src_a_odd);
          if (g_low == i && tid == j) {
            a_even = ue;
            a_odd = uo;
          }
        }
      }
      unsigned a0 = (g_high == 0) ? a_even : 0u;
      unsigned a1 = (g_high == 0) ? a_odd  : 0u;
      unsigned a2 = (g_high == 1) ? a_even : 0u;
      unsigned a3 = (g_high == 1) ? a_odd  : 0u;

      int src_b0 = 4 * tile + 2 * g_high;
      int src_b1 = 4 * tile + 2 * g_high + 1;
      unsigned raw_b0 = 0u, raw_b1 = 0u;
      #pragma unroll
      for (int s = 0; s < 2; s++) {
        #pragma unroll
        for (int j = 0; j < 3; j++) {
          unsigned vb0 = shfl_u32(v_pk[3*s+j], src_b0);
          unsigned vb1 = shfl_u32(v_pk[3*s+j], src_b1);
          if (spin_col == s && tid == j) {
            raw_b0 = vb0;
            raw_b1 = vb1;
          }
        }
      }
      unsigned b0 = 0u, b1 = 0u;
      if (tid < 3) {
        if constexpr (MODE == 1 || MODE == 2) {
          b0 = format_b_16x2(raw_b0, is_im_col);
          b1 = format_b_16x2(raw_b1, is_im_col);
        } else if constexpr (MODE == 4 || MODE == 5 || MODE == 6) {
          b0 = format_b_fp8x2(raw_b0, is_im_col);
          b1 = format_b_fp8x2(raw_b1, is_im_col);
        } else if constexpr (MODE == 7) {
          b0 = format_b_fp4x2(raw_b0, is_im_col);
          b1 = format_b_fp4x2(raw_b1, is_im_col);
        }
      }

      float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      if constexpr (MODE == 1) {
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 2) {
        asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 4) {
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 5) {
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 6) {
        unsigned sfa = 0x7f7f7f7fu; // ue8m0 neutral scale 2^0 = 1.0
        unsigned sfb = 0x7f7f7f7fu;
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
      } else if constexpr (MODE == 7) {
        unsigned sfa = 0x38383838u; // ue4m3 neutral scale 1.0 (exp=7 -> 7<<3 = 0x38)
        unsigned sfb = 0x38383838u;
        asm volatile(
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
          : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
      }

      float l_re = (tid < 2) ? c[0] : c[2];
      float l_im = (tid < 2) ? c[1] : c[3];
      #pragma unroll
      for (int s = 0; s < 2; s++) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
          int sl = (tid & 1) * 16 + 4 * i + (tid & 2) + s;
          float rr = shfl_f32(l_re, sl);
          float ri = shfl_f32(l_im, sl);
          if (group_id == tile) {
            out_re[s*3+i] = rr;
            out_im[s*3+i] = ri;
          }
        }
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; s++)
      #pragma unroll
      for (int i = 0; i < 3; i++)
        out.c[s][i] = {out_re[s*3+i] * unscale, out_im[s*3+i] * unscale};
    return out;
  }
}

template <int MODE>
__global__ void kernel_contract_probe(const SU3Link *__restrict__ links,
                                      const HalfSpinor *__restrict__ in_spinors,
                                      HalfSpinor *__restrict__ out_spinors,
                                      int n_sites) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_sites) return;
  out_spinors[idx] = contract_su3_halfspinor<MODE>(links[idx], in_spinors[idx]);
}

template <int MODE>
__global__ void kernel_onchip_mma_peak(float *out, int iters) {
  unsigned a0 = 0x38383838u, a1 = 0x38383838u, a2 = 0x38383838u, a3 = 0x38383838u;
  unsigned b0 = 0x38383838u, b1 = 0x38383838u;
  unsigned sfa = (MODE == 7) ? 0x38383838u : 0x7f7f7f7fu;
  unsigned sfb = (MODE == 7) ? 0x38383838u : 0x7f7f7f7fu;
  float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
  for (int it = 0; it < iters; it++) {
    #pragma unroll 16
    for (int u = 0; u < 16; u++) {
      if constexpr (MODE == 1) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 2) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 3) {
        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 4) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 5) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if constexpr (MODE == 6) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
      } else if constexpr (MODE == 7) {
        asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                     "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};"
                     : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb));
      }
    }
  }
  if (c0 == -123456.0f) out[threadIdx.x] = c0 + c1 + c2 + c3;
}

static SU3Link random_su3(std::mt19937_64 &rng) {
  std::normal_distribution<double> dist(0.0, 1.0);
  double vr[3][3], vi[3][3];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      vr[i][j] = dist(rng);
      vi[i][j] = dist(rng);
    }
  // Gram-Schmidt row 0
  double n0 = std::sqrt(vr[0][0]*vr[0][0] + vi[0][0]*vi[0][0] +
                        vr[0][1]*vr[0][1] + vi[0][1]*vi[0][1] +
                        vr[0][2]*vr[0][2] + vi[0][2]*vi[0][2]);
  for (int j = 0; j < 3; j++) { vr[0][j] /= n0; vi[0][j] /= n0; }
  // Orthogonalize row 1 to row 0
  double dot_re = 0.0, dot_im = 0.0;
  for (int j = 0; j < 3; j++) {
    dot_re += vr[1][j]*vr[0][j] + vi[1][j]*vi[0][j];
    dot_im += vi[1][j]*vr[0][j] - vr[1][j]*vi[0][j];
  }
  for (int j = 0; j < 3; j++) {
    vr[1][j] -= dot_re*vr[0][j] - dot_im*vi[0][j];
    vi[1][j] -= dot_re*vi[0][j] + dot_im*vr[0][j];
  }
  double n1 = std::sqrt(vr[1][0]*vr[1][0] + vi[1][0]*vi[1][0] +
                        vr[1][1]*vr[1][1] + vi[1][1]*vi[1][1] +
                        vr[1][2]*vr[1][2] + vi[1][2]*vi[1][2]);
  for (int j = 0; j < 3; j++) { vr[1][j] /= n1; vi[1][j] /= n1; }
  // Row 2 = conj(row0 x row1)
  for (int j = 0; j < 3; j++) {
    int j1 = (j + 1) % 3, j2 = (j + 2) % 3;
    double cr = (vr[0][j1]*vr[1][j2] - vi[0][j1]*vi[1][j2]) - (vr[0][j2]*vr[1][j1] - vi[0][j2]*vi[1][j1]);
    double ci = (vr[0][j1]*vi[1][j2] + vi[0][j1]*vr[1][j2]) - (vr[0][j2]*vi[1][j1] + vi[0][j2]*vr[1][j1]);
    vr[2][j] = cr;
    vi[2][j] = -ci;
  }
  SU3Link U;
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      U.e[i][j] = {(float)vr[i][j], (float)vi[i][j]};
  return U;
}

int main() {
  const int n_sites = 65536; // 16^3x32 checkerboard sites
  std::mt19937_64 rng(20261020ULL);
  std::normal_distribution<float> sdist(0.0f, 1.0f);

  std::vector<SU3Link> h_links(n_sites);
  std::vector<HalfSpinor> h_in(n_sites), h_out(n_sites);
  for (int idx = 0; idx < n_sites; idx++) {
    h_links[idx] = random_su3(rng);
    float scale = (idx % 2 == 0) ? 1.0f : 0.03f; // test wide dynamic range across sites
    for (int s = 0; s < 2; s++)
      for (int c = 0; c < 3; c++)
        h_in[idx].c[s][c] = {sdist(rng) * scale, sdist(rng) * scale};
  }

  // Exact FP64 host reference
  std::vector<double> ref_re(n_sites * 6), ref_im(n_sites * 6);
  double ref_norm2 = 0.0;
  for (int idx = 0; idx < n_sites; idx++) {
    for (int s = 0; s < 2; s++) {
      for (int i = 0; i < 3; i++) {
        double re = 0.0, im = 0.0;
        for (int j = 0; j < 3; j++) {
          double ur = h_links[idx].e[i][j].re, ui = h_links[idx].e[i][j].im;
          double vr = h_in[idx].c[s][j].re, vi = h_in[idx].c[s][j].im;
          re += ur * vr - ui * vi;
          im += ur * vi + ui * vr;
        }
        ref_re[idx * 6 + s * 3 + i] = re;
        ref_im[idx * 6 + s * 3 + i] = im;
        ref_norm2 += re * re + im * im;
      }
    }
  }

  SU3Link *d_links = nullptr;
  HalfSpinor *d_in = nullptr, *d_out = nullptr;
  float *d_dummy = nullptr;
  cudaMalloc(&d_links, n_sites * sizeof(SU3Link));
  cudaMalloc(&d_in, n_sites * sizeof(HalfSpinor));
  cudaMalloc(&d_out, n_sites * sizeof(HalfSpinor));
  cudaMalloc(&d_dummy, 1024 * sizeof(float));
  cudaMemcpy(d_links, h_links.data(), n_sites * sizeof(SU3Link), cudaMemcpyHostToDevice);
  cudaMemcpy(d_in, h_in.data(), n_sites * sizeof(HalfSpinor), cudaMemcpyHostToDevice);

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);

  auto eval_mode = [&](const char *name, auto launch_contract, auto launch_onchip, double ops_per_mma) {
    cudaMemset(d_out, 0, n_sites * sizeof(HalfSpinor));
    launch_contract(1);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out.data(), d_out, n_sites * sizeof(HalfSpinor), cudaMemcpyDeviceToHost);

    double err_norm2 = 0.0, max_abs_err = 0.0;
    for (int idx = 0; idx < n_sites; idx++) {
      for (int s = 0; s < 2; s++) {
        for (int i = 0; i < 3; i++) {
          double dr = (double)h_out[idx].c[s][i].re - ref_re[idx * 6 + s * 3 + i];
          double di = (double)h_out[idx].c[s][i].im - ref_im[idx * 6 + s * 3 + i];
          err_norm2 += dr * dr + di * di;
          max_abs_err = std::max(max_abs_err, std::hypot(dr, di));
        }
      }
    }
    double rel_frob_err = std::sqrt(err_norm2 / ref_norm2);

    const int n_rep = 1000;
    cudaEventRecord(ev0);
    launch_contract(n_rep);
    cudaEventRecord(ev1);
    cudaEventSynchronize(ev1);
    float ms_contract = 0.0f;
    cudaEventElapsedTime(&ms_contract, ev0, ev1);
    double us_per_call = (ms_contract * 1000.0) / n_rep;
    double ns_per_contraction = (ms_contract * 1e6) / ((double)n_rep * n_sites);
    double gflops_useful = ((double)n_rep * n_sites * 72.0) / (ms_contract * 1e-3) / 1e9;

    double onchip_tflops = 0.0;
    if (ops_per_mma > 0.0) {
      launch_onchip(64);
      cudaDeviceSynchronize();
      const int iters = 4096;
      cudaEventRecord(ev0);
      launch_onchip(iters);
      cudaEventRecord(ev1);
      cudaEventSynchronize(ev1);
      float ms_onchip = 0.0f;
      cudaEventElapsedTime(&ms_onchip, ev0, ev1);
      double total_mma = 160.0 * (256.0 / 32.0) * iters * 16.0;
      onchip_tflops = (total_mma * ops_per_mma) / (ms_onchip * 1e-3) / 1e12;
    }

    std::printf("MODE=%-24s rel_frob_err=%.6e max_abs_err=%.6e stream_us=%.3f ns_per_site=%.4f useful_GFLOPS=%.2f onchip_peak_TFLOPS=%.2f\n",
                name, rel_frob_err, max_abs_err, us_per_call, ns_per_contraction, gflops_useful, onchip_tflops);
  };

  std::printf("=== GB10 (sm_121a) Native Low-Precision MMA & Fixed-Point SU(3)xHalfSpinor Micro-Probe (n_sites=%d) ===\n", n_sites);

  eval_mode("0_FP32_SIMT",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<0><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int){}, 0.0);
  eval_mode("1_FP16_HMMA_16816",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<1><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int it){ kernel_onchip_mma_peak<1><<<160,256>>>(d_dummy,it); }, 4096.0);
  eval_mode("2_BF16_HMMA_16816",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<2><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int it){ kernel_onchip_mma_peak<2><<<160,256>>>(d_dummy,it); }, 4096.0);
  eval_mode("3_TF32_HMMA_1688",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<3><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int it){ kernel_onchip_mma_peak<3><<<160,256>>>(d_dummy,it); }, 2048.0);
  eval_mode("4_FP8_E4M3_QMMA_16832",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<4><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int it){ kernel_onchip_mma_peak<4><<<160,256>>>(d_dummy,it); }, 8192.0);
  eval_mode("5_FP8_E5M2_QMMA_16832",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<5><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int it){ kernel_onchip_mma_peak<5><<<160,256>>>(d_dummy,it); }, 8192.0);
  eval_mode("6_MXFP8_E4M3_QMMA_SF",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<6><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int it){ kernel_onchip_mma_peak<6><<<160,256>>>(d_dummy,it); }, 8192.0);
  eval_mode("7_NVFP4_E2M1_OMMA_SF",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<7><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int it){ kernel_onchip_mma_peak<7><<<160,256>>>(d_dummy,it); }, 16384.0);
  eval_mode("8_INT8_QUARTER_FIXED",
            [&](int R){ for(int r=0;r<R;r++) kernel_contract_probe<8><<<n_sites/256,256>>>(d_links,d_in,d_out,n_sites); },
            [&](int){}, 0.0);

  cudaFree(d_links);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_dummy);
  return 0;
}
