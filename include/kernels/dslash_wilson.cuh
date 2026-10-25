#pragma once

#include <dslash_helper.cuh>
#include <color_spinor_field_order.h>
#include <gauge_field_order.h>
#include <color_spinor.h>
#include <index_helper.cuh>
#include <kernels/dslash_pack.cuh> // for the packing kernel
#include <kernels/spinor_reweight.cuh>
#include <cuda_fp16.h>

#ifndef QUDA_WILSON_MMA_MODE
#define QUDA_WILSON_MMA_MODE 1
#endif

#ifndef QUDA_WILSON_MMA_SMEM
#define QUDA_WILSON_MMA_SMEM 1
#endif

#ifndef QUDA_WILSON_MMA_DIRS
#define QUDA_WILSON_MMA_DIRS 2
#endif

#ifndef QUDA_WILSON_MAX_REGS
#define QUDA_WILSON_MAX_REGS 0
#endif

#ifndef QUDA_ENABLE_TC_COUNTER
#define QUDA_ENABLE_TC_COUNTER 0
#endif

namespace quda
{

#if defined(QUDA_ENABLE_TC_COUNTER) && (QUDA_ENABLE_TC_COUNTER == 1)
  static __device__ unsigned long long quda_tc_hmma_counter_d = 0ULL;
#endif

  __device__ __forceinline__ unsigned tc_shfl_u32(unsigned val, int src_lane)
  {
#if defined(__CUDA_ARCH__)
    unsigned res;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=r"(res) : "r"(val), "r"(src_lane));
    return res;
#else
    (void)src_lane;
    return val;
#endif
  }

  __device__ __forceinline__ float tc_shfl_f32(float val, int src_lane)
  {
#if defined(__CUDA_ARCH__)
    float res;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(res) : "f"(val), "r"(src_lane));
    return res;
#else
    (void)src_lane;
    return val;
#endif
  }

  __device__ __forceinline__ unsigned tc_pack_half2(float lo, float hi)
  {
#if defined(__CUDA_ARCH__)
    __half2 h = __floats2half2_rn(lo, hi);
    return reinterpret_cast<unsigned &>(h);
#else
    (void)lo;
    (void)hi;
    return 0u;
#endif
  }

  __device__ __forceinline__ unsigned tc_pack_bf162(float lo, float hi)
  {
#if defined(__CUDA_ARCH__)
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
#else
    (void)lo;
    (void)hi;
    return 0u;
#endif
  }

  __device__ __forceinline__ unsigned tc_pack_e4m3x2(float lo, float hi)
  {
#if defined(__CUDA_ARCH__)
    unsigned short s;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(s) : "f"(lo), "f"(hi));
    return (unsigned)s;
#else
    (void)lo;
    (void)hi;
    return 0u;
#endif
  }

  __device__ __forceinline__ unsigned tc_pack_e2m1x2(float lo, float hi)
  {
#if defined(__CUDA_ARCH__)
    unsigned short s;
    asm("{ .reg .b8 __t; cvt.rn.satfinite.e2m1x2.f32 __t, %2, %1; mov.b16 %0, {__t, 0}; }" : "=h"(s) : "f"(lo), "f"(hi));
    return (unsigned)(s & 0xffu);
#else
    (void)lo;
    (void)hi;
    return 0u;
#endif
  }

  __device__ __forceinline__ void tc_pack_half2_split(unsigned &hi_u, unsigned &lo_u, float x, float y)
  {
#if defined(__CUDA_ARCH__)
    __half2 h_hi = __floats2half2_rn(x, y);
    float x_hi = __low2float(h_hi);
    float y_hi = __high2float(h_hi);
    __half2 h_lo = __floats2half2_rn((x - x_hi) * 2048.0f, (y - y_hi) * 2048.0f);
    hi_u = reinterpret_cast<unsigned &>(h_hi);
    lo_u = reinterpret_cast<unsigned &>(h_lo);
#else
    (void)x;
    (void)y;
    hi_u = 0u;
    lo_u = 0u;
#endif
  }

  __device__ __forceinline__ unsigned tc_format_b_half2(unsigned pk_re_im, int is_im_col)
  {
    return is_im_col ? ((pk_re_im >> 16) | (pk_re_im << 16)) : (pk_re_im ^ 0x80000000u);
  }

  __device__ __forceinline__ unsigned tc_float_to_tf32_bits(float x)
  {
#if defined(__CUDA_ARCH__)
    unsigned r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    (void)x;
    return 0u;
#endif
  }

  /**
     @brief Warp-collaborative Tensor Core (mma.sync.aligned) SU(3) x HalfVector contraction
     for 32-bit float inner arithmetic (used by sloppy half and single Wilson Dslash).
     Modes:
       1: FP16 m16n8k16 (HMMA.16816.F32)
       3: 3xFP16 split m16n8k16
       4: TF32 m16n8k8 (HMMA.1688.F32.TF32)
       5: BF16 m16n8k16 (HMMA.16816.F32.BF16)
       6: FP8 E4M3 m16n8k32 (QMMA.16832.F32.E4M3.E4M3)
       7: 8-bit Quarter Fixed-Point (int8_t, round(127*x)) sloppy contraction control
       8: NVFP4 E2M1 m16n8k64 block-scaled (OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X)
   */
  __device__ __host__ __forceinline__ ColorSpinor<float, 3, 2>
  mma_su3_mul_halfvec(const Matrix<complex<float>, 3> &U, const ColorSpinor<float, 3, 2> &in, bool has_hop)
  {
#if defined(__CUDA_ARCH__)
#if QUDA_WILSON_MMA_MODE == 7
    // Mode 7: 8-bit Quarter Fixed-Point (int8_t, round(127 * x)) sloppy contraction control
    ColorSpinor<float, 3, 2> out_vec;
    if (!has_hop) {
#pragma unroll
      for (int idx = 0; idx < 6; idx++) out_vec.data[idx] = complex<float>(0.0f, 0.0f);
      return out_vec;
    }
    float my_vmax = 0.0f;
#pragma unroll
    for (int idx = 0; idx < 6; idx++) {
      my_vmax = fmaxf(my_vmax, fabsf(in.data[idx].real()));
      my_vmax = fmaxf(my_vmax, fabsf(in.data[idx].imag()));
    }
    const float v_scale = (my_vmax > 0.0f) ? (127.0f / my_vmax) : 0.0f;
    const float unscale = (my_vmax > 0.0f) ? (my_vmax * (1.0f / (127.0f * 127.0f))) : 0.0f;
    int u_q_re[3][3], u_q_im[3][3];
    int v_q_re[2][3], v_q_im[2][3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        u_q_re[i][j] = __float2int_rn(fminf(fmaxf(U(i, j).real() * 127.0f, -127.0f), 127.0f));
        u_q_im[i][j] = __float2int_rn(fminf(fmaxf(U(i, j).imag() * 127.0f, -127.0f), 127.0f));
      }
    }
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        v_q_re[s][j] = __float2int_rn(fminf(fmaxf(in.data[s * 3 + j].real() * v_scale, -127.0f), 127.0f));
        v_q_im[s][j] = __float2int_rn(fminf(fmaxf(in.data[s * 3 + j].imag() * v_scale, -127.0f), 127.0f));
      }
    }
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int i = 0; i < 3; i++) {
        int acc_re = 0, acc_im = 0;
#pragma unroll
        for (int j = 0; j < 3; j++) {
          acc_re += u_q_re[i][j] * v_q_re[s][j] - u_q_im[i][j] * v_q_im[s][j];
          acc_im += u_q_re[i][j] * v_q_im[s][j] + u_q_im[i][j] * v_q_re[s][j];
        }
        out_vec.data[s * 3 + i] = complex<float>((float)acc_re * unscale, (float)acc_im * unscale);
      }
    }
    return out_vec;
#else
    int lane_id;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    const int group_id = lane_id >> 2; // 0..7
    const int tid = lane_id & 3;       // 0..3
    const int g_low = group_id & 3;    // 0..3
    const int g_high = group_id >> 2;  // 0..1

#if defined(QUDA_WILSON_MMA_SMEM) && (QUDA_WILSON_MMA_SMEM == 1) && (QUDA_WILSON_MMA_MODE == 8)
    __shared__ __align__(8) unsigned quda_tc_smem[8][32][12];
    const int warp_id = ((threadIdx.z * blockDim.y + threadIdx.y) * (blockDim.x >> 5) + (threadIdx.x >> 5)) & 7;
    unsigned (*slot)[12] = quda_tc_smem[warp_id];
    unsigned char (*slot_u8)[48] = reinterpret_cast<unsigned char (*)[48]>(slot);

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float u_re = has_hop ? U(i, j).real() : 0.0f;
        float u_im = has_hop ? U(i, j).imag() : 0.0f;
        slot_u8[lane_id][3 * i + j] = static_cast<unsigned char>(tc_pack_e2m1x2(u_re * 6.0f, u_im * 6.0f));
      }
    }

    float my_vmax = 0.0f;
    if (has_hop) {
#pragma unroll
      for (int idx = 0; idx < 6; idx++) {
        float ar = fabsf(in.data[idx].real());
        float ai = fabsf(in.data[idx].imag());
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(my_vmax) : "f"(my_vmax), "f"(ar));
        asm("max.NaN.f32 %0, %1, %2;" : "=f"(my_vmax) : "f"(my_vmax), "f"(ai));
      }
    }
    const unsigned vmax_exp = (__float_as_uint(my_vmax) >> 23) & 0xffu;
    float inv_vmax = 0.0f;
    float unscale_vmax = 0.0f;
    if (vmax_exp > 0u && vmax_exp < 255u) {
      asm("div.rn.f32 %0, %1, %2;" : "=f"(inv_vmax) : "f"(6.0f), "f"(my_vmax));
      unscale_vmax = my_vmax;
    } else if (vmax_exp == 255u) {
      unscale_vmax = __int_as_float(0x7fffffff);
    }

#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float v_re = has_hop ? (in.data[s * 3 + j].real() * inv_vmax) : 0.0f;
        float v_im = has_hop ? (in.data[s * 3 + j].imag() * inv_vmax) : 0.0f;
        unsigned pk = tc_pack_e2m1x2(v_re, v_im);
        slot_u8[lane_id][9 + 3 * s + j] = static_cast<unsigned char>(pk ^ 0x80u);
        slot_u8[lane_id][15 + 3 * s + j] = static_cast<unsigned char>(((pk >> 4) | (pk << 4)) & 0xffu);
      }
    }
    slot_u8[lane_id][21] = 0u;
    __syncwarp();

    const bool a_valid = (g_low < 3) && (tid < 3);
    const int u_idx = a_valid ? (3 * g_low + tid) : 21;
    const int spin_col = (g_low >> 1) & 1;
    const int is_im_col = g_low & 1;
    const int v_idx = (tid < 3) ? (9 + 6 * is_im_col + 3 * spin_col + tid) : 21;
    const int dst_q_top = g_high | ((tid & 2) << 1);
    const int dst_q_bot = dst_q_top | 2;
    const int dst_comp = 3 * (tid & 1) + g_low;
    const unsigned b_shift = 8u * static_cast<unsigned>(g_high);

    const unsigned char * __restrict__ a_ptr = &slot_u8[g_high][u_idx];
    const unsigned char * __restrict__ b_ptr = &slot_u8[4 * g_high][v_idx];
    float2 * __restrict__ c_ptr_top = &reinterpret_cast<float2 *>(slot[dst_q_top])[dst_comp];
    float2 * __restrict__ c_ptr_bot = &reinterpret_cast<float2 *>(slot[dst_q_bot])[dst_comp];
    const float zero_f = 0.0f;
    const unsigned sf_neutral = 0x38383838u;

    float c_top_re[4], c_top_im[4], c_bot_re[4], c_bot_im[4];

#pragma unroll
    for (int tile = 0; tile < 4; tile++) {
      const unsigned a_s0 = a_ptr[tile * 384];
      const unsigned a_s4 = a_ptr[tile * 384 + 192];
      const unsigned a_s2 = a_ptr[tile * 384 + 96];
      const unsigned a_s6 = a_ptr[tile * 384 + 288];
      const unsigned a_first  = a_s0 | (a_s4 << 8);
      const unsigned a_second = (a_s2 << 16) | (a_s6 << 24);
      const unsigned a0 = (g_high == 0) ? a_first  : 0u;
      const unsigned a1 = (g_high == 0) ? a_second : 0u;
      const unsigned a2 = (g_high == 1) ? a_first  : 0u;
      const unsigned a3 = (g_high == 1) ? a_second : 0u;

      const unsigned b_s0 = b_ptr[tile * 384];
      const unsigned b_s2 = b_ptr[tile * 384 + 96];
      const unsigned b_s1 = b_ptr[tile * 384 + 48];
      const unsigned b_s3 = b_ptr[tile * 384 + 144];
      const unsigned b0 = (b_s0 | (b_s2 << 16)) << b_shift;
      const unsigned b1 = (b_s1 | (b_s3 << 16)) << b_shift;

      float c0, c1, c2, c3;
      asm(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, %11, {0, 0}, %11, {0, 0};"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(zero_f), "r"(sf_neutral));

      c_top_re[tile] = c0;
      c_top_im[tile] = c1;
      c_bot_re[tile] = c2;
      c_bot_im[tile] = c3;
    }

    if (g_low < 3) {
#pragma unroll
      for (int tile = 0; tile < 4; tile++) {
        c_ptr_top[tile * 48] = make_float2(c_top_re[tile], c_top_im[tile]);
        c_ptr_bot[tile * 48] = make_float2(c_bot_re[tile], c_bot_im[tile]);
      }
    }
    __syncwarp();

#if defined(QUDA_ENABLE_TC_COUNTER) && (QUDA_ENABLE_TC_COUNTER == 1)
    if (lane_id == 0) {
      atomicAdd(&quda_tc_hmma_counter_d, 4ULL);
    }
#endif

    ColorSpinor<float, 3, 2> out_vec;
    const float2 *my_out = reinterpret_cast<const float2 *>(slot[lane_id]);
    const float final_scale = unscale_vmax * (1.0f / 36.0f);
#pragma unroll
    for (int idx = 0; idx < 6; idx++) {
      float2 v = my_out[idx];
      out_vec.data[idx] = complex<float>(v.x * final_scale, v.y * final_scale);
    }
    return out_vec;
#elif defined(QUDA_WILSON_MMA_SMEM) && (QUDA_WILSON_MMA_SMEM == 1) && (QUDA_WILSON_MMA_MODE == 1 || QUDA_WILSON_MMA_MODE == 5 || QUDA_WILSON_MMA_MODE == 6)
    __shared__ __align__(8) unsigned quda_tc_smem[4][32][24];
    const int warp_id = ((threadIdx.z * blockDim.y + threadIdx.y) * (blockDim.x >> 5) + (threadIdx.x >> 5)) & 3;
    unsigned (*slot)[24] = quda_tc_smem[warp_id];

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float u_re = has_hop ? U(i, j).real() : 0.0f;
        float u_im = has_hop ? U(i, j).imag() : 0.0f;
#if QUDA_WILSON_MMA_MODE == 1
        slot[lane_id][3 * i + j] = tc_pack_half2(u_re, u_im);
#elif QUDA_WILSON_MMA_MODE == 5
        slot[lane_id][3 * i + j] = tc_pack_bf162(u_re, u_im);
#elif QUDA_WILSON_MMA_MODE == 6
        slot[lane_id][3 * i + j] = tc_pack_e4m3x2(u_re, u_im);
#endif
      }
    }

    float my_vmax = 0.0f;
    if (has_hop) {
#pragma unroll
      for (int idx = 0; idx < 6; idx++) {
        my_vmax = fmaxf(my_vmax, fabsf(in.data[idx].real()));
        my_vmax = fmaxf(my_vmax, fabsf(in.data[idx].imag()));
      }
    }
    const unsigned vmax_exp = __float_as_uint(my_vmax) & 0x7f800000u;
    const float inv_vmax = (vmax_exp > 0u) ? __uint_as_float(0x7f000000u - vmax_exp) : 0.0f;
    const float unscale_vmax = (vmax_exp > 0u) ? __uint_as_float(vmax_exp) : 0.0f;

#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float v_re = has_hop ? (in.data[s * 3 + j].real() * inv_vmax) : 0.0f;
        float v_im = has_hop ? (in.data[s * 3 + j].imag() * inv_vmax) : 0.0f;
#if QUDA_WILSON_MMA_MODE == 1
        unsigned pk = tc_pack_half2(v_re, v_im);
        slot[lane_id][9 + 3 * s + j] = pk ^ 0x80000000u;
        slot[lane_id][15 + 3 * s + j] = (pk >> 16) | (pk << 16);
#elif QUDA_WILSON_MMA_MODE == 5
        unsigned pk = tc_pack_bf162(v_re, v_im);
        slot[lane_id][9 + 3 * s + j] = pk ^ 0x80000000u;
        slot[lane_id][15 + 3 * s + j] = (pk >> 16) | (pk << 16);
#elif QUDA_WILSON_MMA_MODE == 6
        unsigned pk = tc_pack_e4m3x2(v_re, v_im);
        unsigned b0_b = pk & 0xffu;
        unsigned b1_b = (pk >> 8) & 0xffu;
        slot[lane_id][9 + 3 * s + j] = b0_b | ((b1_b ^ 0x80u) << 8);
        slot[lane_id][15 + 3 * s + j] = b1_b | (b0_b << 8);
#endif
      }
    }
    slot[lane_id][21] = 0u;
    __syncwarp();

    const bool a_valid = (g_low < 3) && (tid < 3);
    const int u_idx = a_valid ? (3 * g_low + tid) : 21;
    const int spin_col = (g_low >> 1) & 1;
    const int is_im_col = g_low & 1;
    const int v_idx = (tid < 3) ? (9 + 6 * is_im_col + 3 * spin_col + tid) : 21;
    const int dst_q = g_high | (tid & 2);
    const int dst_comp = 3 * (tid & 1) + g_low;

    const unsigned * __restrict__ a_ptr = &slot[g_high][u_idx];
    const unsigned * __restrict__ b_ptr = &slot[2 * g_high][v_idx];
    float2 * __restrict__ c_ptr = &reinterpret_cast<float2 *>(slot[dst_q])[dst_comp];
    const float zero_f = 0.0f;

    float c_re_arr[8];
    float c_im_arr[8];

#pragma unroll
    for (int tile = 0; tile < 8; tile++) {
      const unsigned a_even = a_ptr[tile * 96];
      const unsigned a_odd  = a_ptr[tile * 96 + 48];
      const unsigned a0 = (g_high == 0) ? a_even : 0u;
      const unsigned a1 = (g_high == 0) ? a_odd  : 0u;
      const unsigned a2 = (g_high == 1) ? a_even : 0u;
      const unsigned a3 = (g_high == 1) ? a_odd  : 0u;

      const unsigned b0 = b_ptr[tile * 96];
      const unsigned b1 = b_ptr[tile * 96 + 24];

      float c0, c1, c2, c3;
#if QUDA_WILSON_MMA_MODE == 1
      asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(zero_f));
#elif QUDA_WILSON_MMA_MODE == 5
      asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(zero_f));
#elif QUDA_WILSON_MMA_MODE == 6
      asm(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
        : "=f"(c0), "=f"(c1), "=f"(c2), "=f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(zero_f));
#endif

      c_re_arr[tile] = (tid < 2) ? c0 : c2;
      c_im_arr[tile] = (tid < 2) ? c1 : c3;
    }

    if (g_low < 3) {
#pragma unroll
      for (int tile = 0; tile < 8; tile++) {
        c_ptr[tile * 48] = make_float2(c_re_arr[tile], c_im_arr[tile]);
      }
    }
    __syncwarp();

#if defined(QUDA_ENABLE_TC_COUNTER) && (QUDA_ENABLE_TC_COUNTER == 1)
    if (lane_id == 0) {
      atomicAdd(&quda_tc_hmma_counter_d, 8ULL);
    }
#endif

    ColorSpinor<float, 3, 2> out_vec;
    const float2 *my_out = reinterpret_cast<const float2 *>(slot[lane_id]);
    const float final_scale = unscale_vmax;
#pragma unroll
    for (int idx = 0; idx < 6; idx++) {
      float2 v = my_out[idx];
      out_vec.data[idx] = complex<float>(v.x * final_scale, v.y * final_scale);
    }
    return out_vec;
#else
    float out_re[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float out_im[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#if QUDA_WILSON_MMA_MODE == 4
    // Mode 4: 1xTF32 (mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32), 2 sites per instruction
    unsigned u_tf32[3][6];
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float re = has_hop ? U(i, j).real() : 0.0f;
        float im = has_hop ? U(i, j).imag() : 0.0f;
        u_tf32[i][2 * j + 0] = tc_float_to_tf32_bits(re);
        u_tf32[i][2 * j + 1] = tc_float_to_tf32_bits(im);
      }
    }

    unsigned v_tf32[2][6];
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float re = has_hop ? in.data[s * 3 + j].real() : 0.0f;
        float im = has_hop ? in.data[s * 3 + j].imag() : 0.0f;
        v_tf32[s][2 * j + 0] = tc_float_to_tf32_bits(re);
        v_tf32[s][2 * j + 1] = tc_float_to_tf32_bits(im);
      }
    }

    const int spin_col = (g_low >> 1) & 1;
    const int is_im_col = g_low & 1;
    const int pair_id = lane_id >> 1; // 0..15
    const int q_pair = lane_id & 1;   // 0..1

#pragma unroll
    for (int tile = 0; tile < 16; tile++) {
      const int src_s0 = 2 * tile + 0;
      const int src_s1 = 2 * tile + 1;

      unsigned a0 = 0u, a1 = 0u, a2 = 0u, a3 = 0u;
#pragma unroll
      for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int k = 0; k < 6; k++) {
          unsigned u0 = tc_shfl_u32(u_tf32[i][k], src_s0);
          unsigned u1 = tc_shfl_u32(u_tf32[i][k], src_s1);
          const bool m_i = (group_id == i);
          const bool m_k0 = m_i && (tid == k);
          const bool m_k1 = m_i && (tid + 4 == k);
          a0 = m_k0 ? u0 : a0;
          a1 = m_k0 ? u1 : a1;
          a2 = m_k1 ? u0 : a2;
          a3 = m_k1 ? u1 : a3;
        }
      }

      const int src_b = (g_high == 0) ? src_s0 : src_s1;
      unsigned b0 = 0u, b1 = 0u;
#pragma unroll
      for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
          unsigned vr = tc_shfl_u32(v_tf32[s][2 * j + 0], src_b);
          unsigned vi = tc_shfl_u32(v_tf32[s][2 * j + 1], src_b);
          const bool m_s = (spin_col == s);
          unsigned k_even = is_im_col ? vi : vr;
          unsigned k_odd = is_im_col ? vr : (vi ^ 0x80000000u);
          b0 = (m_s && (tid == 2 * j + 0)) ? k_even : b0;
          b0 = (m_s && (tid == 2 * j + 1)) ? k_odd : b0;
          b1 = (m_s && (tid + 4 == 2 * j + 0)) ? k_even : b1;
          b1 = (m_s && (tid + 4 == 2 * j + 1)) ? k_odd : b1;
        }
      }

      float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

      const float lane_out_re = (tid < 2) ? c[0] : c[2];
      const float lane_out_im = (tid < 2) ? c[1] : c[3];
      const bool is_my_tile = (pair_id == tile);
#pragma unroll
      for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int i = 0; i < 3; i++) {
          const int src_out_lane = 4 * i + 2 * q_pair + s;
          float r_re = tc_shfl_f32(lane_out_re, src_out_lane);
          float r_im = tc_shfl_f32(lane_out_im, src_out_lane);
          out_re[s * 3 + i] = is_my_tile ? r_re : out_re[s * 3 + i];
          out_im[s * 3 + i] = is_my_tile ? r_im : out_im[s * 3 + i];
        }
      }
    }
#if defined(QUDA_ENABLE_TC_COUNTER) && (QUDA_ENABLE_TC_COUNTER == 1)
    if (lane_id == 0) {
      atomicAdd(&quda_tc_hmma_counter_d, 16ULL);
    }
#endif

#else
    // Mode 1 (1xFP16) and Mode 3 (3xFP16): mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32, 4 sites per instruction
    float my_vmax = 0.0f;
    if (has_hop) {
#pragma unroll
      for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int c = 0; c < 3; c++) {
          my_vmax = fmaxf(my_vmax, fabsf(in.data[s * 3 + c].real()));
          my_vmax = fmaxf(my_vmax, fabsf(in.data[s * 3 + c].imag()));
        }
      }
    }
    const float inv_vmax = (my_vmax > 0.0f) ? (1.0f / my_vmax) : 0.0f;

    unsigned u_hi[9];
    unsigned v_hi[6];
#if QUDA_WILSON_MMA_MODE == 3
    unsigned u_lo[9];
    unsigned v_lo[6];
#endif

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float u_re = has_hop ? U(i, j).real() : 0.0f;
        float u_im = has_hop ? U(i, j).imag() : 0.0f;
#if QUDA_WILSON_MMA_MODE == 3
        tc_pack_half2_split(u_hi[3 * i + j], u_lo[3 * i + j], u_re, u_im);
#else
        u_hi[3 * i + j] = tc_pack_half2(u_re, u_im);
#endif
      }
    }

#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
      for (int j = 0; j < 3; j++) {
        float v_re = has_hop ? (in.data[s * 3 + j].real() * inv_vmax) : 0.0f;
        float v_im = has_hop ? (in.data[s * 3 + j].imag() * inv_vmax) : 0.0f;
#if QUDA_WILSON_MMA_MODE == 3
        tc_pack_half2_split(v_hi[3 * s + j], v_lo[3 * s + j], v_re, v_im);
#else
        v_hi[3 * s + j] = tc_pack_half2(v_re, v_im);
#endif
      }
    }

    const int spin_col = (g_low >> 1) & 1;
    const int is_im_col = g_low & 1;

#pragma unroll
    for (int tile = 0; tile < 8; tile++) {
      const int src_a_even = 4 * tile + g_high;
      const int src_a_odd = 4 * tile + g_high + 2;

      unsigned a_even_hi = 0u, a_odd_hi = 0u;
#if QUDA_WILSON_MMA_MODE == 3
      unsigned a_even_lo = 0u, a_odd_lo = 0u;
#endif
#pragma unroll
      for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
          unsigned ue_hi = tc_shfl_u32(u_hi[3 * i + j], src_a_even);
          unsigned uo_hi = tc_shfl_u32(u_hi[3 * i + j], src_a_odd);
          const bool match_ij = (g_low == i) && (tid == j);
          a_even_hi = match_ij ? ue_hi : a_even_hi;
          a_odd_hi = match_ij ? uo_hi : a_odd_hi;
#if QUDA_WILSON_MMA_MODE == 3
          unsigned ue_lo = tc_shfl_u32(u_lo[3 * i + j], src_a_even);
          unsigned uo_lo = tc_shfl_u32(u_lo[3 * i + j], src_a_odd);
          a_even_lo = match_ij ? ue_lo : a_even_lo;
          a_odd_lo = match_ij ? uo_lo : a_odd_lo;
#endif
        }
      }

      unsigned a_hi[4];
      a_hi[0] = (g_high == 0) ? a_even_hi : 0u;
      a_hi[1] = (g_high == 0) ? a_odd_hi : 0u;
      a_hi[2] = (g_high == 1) ? a_even_hi : 0u;
      a_hi[3] = (g_high == 1) ? a_odd_hi : 0u;
#if QUDA_WILSON_MMA_MODE == 3
      unsigned a_lo[4];
      a_lo[0] = (g_high == 0) ? a_even_lo : 0u;
      a_lo[1] = (g_high == 0) ? a_odd_lo : 0u;
      a_lo[2] = (g_high == 1) ? a_even_lo : 0u;
      a_lo[3] = (g_high == 1) ? a_odd_lo : 0u;
#endif

      const int src_b0 = 4 * tile + 2 * g_high;
      const int src_b1 = 4 * tile + 2 * g_high + 1;

      unsigned raw_b0_hi = 0u, raw_b1_hi = 0u;
#if QUDA_WILSON_MMA_MODE == 3
      unsigned raw_b0_lo = 0u, raw_b1_lo = 0u;
#endif
#pragma unroll
      for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
          unsigned vb0_hi = tc_shfl_u32(v_hi[3 * s + j], src_b0);
          unsigned vb1_hi = tc_shfl_u32(v_hi[3 * s + j], src_b1);
          const bool match_sj = (spin_col == s) && (tid == j);
          raw_b0_hi = match_sj ? vb0_hi : raw_b0_hi;
          raw_b1_hi = match_sj ? vb1_hi : raw_b1_hi;
#if QUDA_WILSON_MMA_MODE == 3
          unsigned vb0_lo = tc_shfl_u32(v_lo[3 * s + j], src_b0);
          unsigned vb1_lo = tc_shfl_u32(v_lo[3 * s + j], src_b1);
          raw_b0_lo = match_sj ? vb0_lo : raw_b0_lo;
          raw_b1_lo = match_sj ? vb1_lo : raw_b1_lo;
#endif
        }
      }

      unsigned b_hi[2];
      b_hi[0] = (tid < 3) ? tc_format_b_half2(raw_b0_hi, is_im_col) : 0u;
      b_hi[1] = (tid < 3) ? tc_format_b_half2(raw_b1_hi, is_im_col) : 0u;

      float c_hi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c_hi[0]), "+f"(c_hi[1]), "+f"(c_hi[2]), "+f"(c_hi[3])
        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));

#if QUDA_WILSON_MMA_MODE == 3
      unsigned b_lo[2];
      b_lo[0] = (tid < 3) ? tc_format_b_half2(raw_b0_lo, is_im_col) : 0u;
      b_lo[1] = (tid < 3) ? tc_format_b_half2(raw_b1_lo, is_im_col) : 0u;

      float c_lo[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c_lo[0]), "+f"(c_lo[1]), "+f"(c_lo[2]), "+f"(c_lo[3])
        : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c_lo[0]), "+f"(c_lo[1]), "+f"(c_lo[2]), "+f"(c_lo[3])
        : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
#pragma unroll
      for (int r = 0; r < 4; r++) {
        c_hi[r] += (1.0f / 2048.0f) * c_lo[r];
      }
#endif

      const float lane_out_re = (tid < 2) ? c_hi[0] : c_hi[2];
      const float lane_out_im = (tid < 2) ? c_hi[1] : c_hi[3];
      const bool is_my_tile = (group_id == tile);
#pragma unroll
      for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int i = 0; i < 3; i++) {
          const int src_out_lane = (tid & 1) * 16 + 4 * i + (tid & 2) + s;
          float r_re = tc_shfl_f32(lane_out_re, src_out_lane);
          float r_im = tc_shfl_f32(lane_out_im, src_out_lane);
          out_re[s * 3 + i] = is_my_tile ? r_re : out_re[s * 3 + i];
          out_im[s * 3 + i] = is_my_tile ? r_im : out_im[s * 3 + i];
        }
      }
    }
#pragma unroll
    for (int idx = 0; idx < 6; idx++) {
      out_re[idx] *= my_vmax;
      out_im[idx] *= my_vmax;
    }
#if defined(QUDA_ENABLE_TC_COUNTER) && (QUDA_ENABLE_TC_COUNTER == 1)
    if (lane_id == 0) {
#if QUDA_WILSON_MMA_MODE == 3
      atomicAdd(&quda_tc_hmma_counter_d, 24ULL);
#else
      atomicAdd(&quda_tc_hmma_counter_d, 8ULL);
#endif
    }
#endif
#endif

    ColorSpinor<float, 3, 2> out_vec;
#pragma unroll
    for (int idx = 0; idx < 6; idx++) {
      out_vec.data[idx] = complex<float>(out_re[idx], out_im[idx]);
    }
    return out_vec;
#endif
#endif
#else
    return has_hop ? (U * in) : ColorSpinor<float, 3, 2>();
#endif
  }

  /**
     @brief Parameter structure for driving the Wilson operator
   */
  template <typename Float, int nColor_, int nDim, typename DDArg, QudaReconstructType reconstruct_, bool distance_pc_ = false>
  struct WilsonArg : DslashArg<Float, nDim, DDArg> {
    static constexpr int nColor = nColor_;
    static constexpr int nSpin = 4;
    static constexpr bool spin_project = true;
    static constexpr bool spinor_direct_load = false; // false means texture load
    static constexpr int max_regs = (sizeof(typename mapper<Float>::type) == sizeof(float) && nColor_ == 3) ? QUDA_WILSON_MAX_REGS : 0;
    typedef typename colorspinor_mapper<Float, nSpin, nColor, spin_project, spinor_direct_load, true>::type F;

    using Ghost = typename colorspinor::GhostNOrder<Float, nSpin, nColor, spin_project, spinor_direct_load, false>;

    static constexpr QudaReconstructType reconstruct = reconstruct_;
    static constexpr bool distance_pc = distance_pc_;
    static constexpr bool gauge_direct_load = false; // false means texture load
    static constexpr QudaGhostExchange ghost = QUDA_GHOST_EXCHANGE_PAD;
    template <bool shifted>
    using G = typename gauge_mapper<Float, reconstruct, 18, QUDA_STAGGERED_PHASE_NO, gauge_direct_load, ghost, false,
                                    QUDA_NATIVE_GAUGE_ORDER, shifted, QUDA_VECTOR_GEOMETRY>::type;

    typedef typename mapper<Float>::type real;

    F out[MAX_MULTI_RHS]; /** output vector field set */
    F in[MAX_MULTI_RHS];  /** input vector field set */
    F x[MAX_MULTI_RHS];   /** input vector set when doing xpay */
    Ghost halo_pack;
    Ghost halo;
    mutable G<false> U;    /** the gauge field */
    mutable G<true> Uback; /** the backwards gauge field */
    const real a; /** xpay scale factor - can be -kappa or -kappa^2 */
    /** parameters for distance preconditioning */
    const real alpha0;
    const int t0;
    static constexpr int prefetch_distance = QUDA_DSLASH_PREFETCH_DISTANCE_WILSON;

    WilsonArg(cvector_ref<ColorSpinorField> &out, cvector_ref<const ColorSpinorField> &in, const ColorSpinorField &halo,
              const GaugeField &U, double a, cvector_ref<const ColorSpinorField> &x, int parity, bool dagger,
              const int *comm_override, double alpha0 = 0.0, int t0 = -1) :
      DslashArg<Float, nDim, DDArg>(out, in, halo, U, x, parity, dagger, a != 0.0 ? true : false, spin_project,
                                    comm_override),
      halo_pack(halo),
      halo(halo),
      U(U),
      Uback(dslash_double_store() ? U.shift(1) : U),
      a(a),
      alpha0(alpha0),
      t0(t0)
    {
      for (auto i = 0u; i < out.size(); i++) {
        this->out[i] = out[i];
        this->in[i] = in[i];
        this->x[i] = x[i];
      }
    }
  };

  /**
     @tparam distance The distance away we are prefetching
     @param[in] dim The dimension we are presently working on
     @param[in] dir The direction we are presently working on (1 = forwards, 0 = backwards)
     @param[in] coord Coordinates that we are working on
     @param[in] parity Partiry that we are working on
     @param[in] arg Paramter struct
  */
  template <class coord_t, class Arg>
  __device__ __host__ void prefetch(int dim, int dir, const coord_t &coord, int parity, const Arg &arg)
  {
    if constexpr (Arg::prefetch_distance == 0) return;

    int step = 2 * dim + dir + Arg::prefetch_distance;
    if (step >= 8) return;

    int dim2 = step / 2;

    // if using a bulk prefetch we need to use block's first coordinate
    auto x_cb = dslash_prefetch_tma() ? coord.x_cb_0 : coord.x_cb;
    x_cb = (Arg::nDim == 5 ? x_cb % arg.dc.volume_4d_cb : x_cb);

    switch (step % 2) {
    case 0: arg.U.template prefetch<Arg::prefetch_type>(x_cb, dim2, parity); break;
    case 1:
      if constexpr (dslash_double_store()) {
        arg.Uback.template prefetch<Arg::prefetch_type>(x_cb, dim2, parity);
      } else {
        int idx = getNeighborIndexCB(coord, dim2, -1, arg.dc);
        arg.U.template prefetch<Arg::prefetch_type>(Arg::nDim == 5 ? idx % arg.dc.volume_4d_cb : idx, dim2, 1 - parity);
      }
      break;
    }
  }

  /**
     @brief Applies the off-diagonal part of the Wilson operator

     @param[out] out The out result field
     @param[in,out] arg Parameter struct
     @param[in] coord Site coordinate struct
     @param[in] s The fifth-dimension index
     @param[in] parity Site parity
     @param[in] idx Thread index (equal to face index for exterior kernels)
     @param[in] thread_dim Which dimension this thread corresponds to (fused exterior only)
  */
  template <bool dagger, KernelType kernel_type, typename Coord, typename Arg, typename Vector>
  __device__ __host__ inline void applyWilson(Vector &out, const Arg &arg, Coord &coord, int parity, int idx,
                                              int thread_dim, bool &active, int src_idx)
  {
    typedef typename mapper<typename Arg::Float>::type real;
    typedef ColorSpinor<real, Arg::nColor, 2> HalfVector;
    typedef Matrix<complex<real>, Arg::nColor> Link;
    const int their_spinor_parity = arg.nParity == 2 ? 1 - parity : 0;

    // parity for gauge field - include residual parity from 5-d => 4-d checkerboarding
    const int gauge_parity = (Arg::nDim == 5 ? (coord.x_cb / arg.dc.volume_4d_cb + parity) % 2 : parity);

    const int t = coord.gx[3];
    const int nt = arg.globalDim3;
    real fwd_coeff_3
      = Arg::distance_pc ? distanceWeight(arg, t + 1, nt) / distanceWeight(arg, t, nt) : static_cast<real>(1.0);
    real bwd_coeff_3
      = Arg::distance_pc ? distanceWeight(arg, t - 1, nt) / distanceWeight(arg, t, nt) : static_cast<real>(1.0);

#pragma unroll
    for (int d = 0; d < 4; d++) { // loop over dimension - 4 and not nDim since this is used for DWF as well
      // Forward gather - compute fwd offset for vector fetch
      if (arg.dd_in.doHopping(coord, d, +1)) {
        const real fwd_coeff = (d < 3) ? 1.0 : fwd_coeff_3;
        const int fwd_idx = getNeighborIndexCB(coord, d, +1, arg.dc);
        const int gauge_idx = (Arg::nDim == 5 ? coord.x_cb % arg.dc.volume_4d_cb : coord.x_cb);
        constexpr int proj_dir = dagger ? +1 : -1;

        const bool ghost = coord.in_boundary[1][d] & isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<1, Arg::nDim>(coord, arg.dc.X, d, arg.nFace) :
            idx;

          Link U = arg.U(d, gauge_idx, gauge_parity);
          HalfVector in = arg.halo.Ghost(d, 1, ghost_idx + (src_idx * arg.Ls + coord.s) * arg.dc.ghostFaceCB[d],
                                         their_spinor_parity);

          out += fwd_coeff * (U * in).reconstruct(d, proj_dir);
        }

        if constexpr (doBulk<kernel_type>()) {
          if constexpr (sizeof(real) == sizeof(float) && Arg::nColor == 3) {
            if ((QUDA_WILSON_MMA_DIRS == 8) || (QUDA_WILSON_MMA_DIRS == 4 && d >= 2) || (QUDA_WILSON_MMA_DIRS == 2 && d == 3)) {
              const bool has_hop = !ghost;
              Link U;
              HalfVector in_proj;
              if (has_hop) {
                U = arg.U(d, gauge_idx, gauge_parity);
                Vector in = arg.in[src_idx](fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
                in_proj = in.project(d, proj_dir);
              }
              HalfVector uv = mma_su3_mul_halfvec(U, in_proj, has_hop);
              if (has_hop) {
                out += fwd_coeff * uv.reconstruct(d, proj_dir);
              }
            } else {
              if (!ghost) {
                Link U = arg.U(d, gauge_idx, gauge_parity);
                Vector in = arg.in[src_idx](fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
                out += fwd_coeff * (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
              }
            }
          } else {
            if (!ghost) {
              Link U = arg.U(d, gauge_idx, gauge_parity);
              Vector in = arg.in[src_idx](fwd_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
              out += fwd_coeff * (U * in.project(d, proj_dir)).reconstruct(d, proj_dir);
            }
          }

          prefetch(d, 0, coord, parity, arg); // prefetch the gauge link Arg::prefetch_distance ahead
        }
      }

      // Backward gather - compute back offset for spinor and gauge fetch
      if (arg.dd_in.doHopping(coord, d, -1)) {
        const real bwd_coeff = (d < 3) ? 1.0 : bwd_coeff_3;
        const int back_idx = getNeighborIndexCB(coord, d, -1, arg.dc);
        int gauge_idx = dslash_double_store() ? coord.x_cb : back_idx;
        if constexpr (Arg::nDim == 5) gauge_idx = gauge_idx % arg.dc.volume_4d_cb;
        constexpr int proj_dir = dagger ? -1 : +1;

        const bool ghost = coord.in_boundary[0][d] & isActive<kernel_type>(active, thread_dim, d, coord, arg);

        if (doHalo<kernel_type>(d) && ghost) {
          // we need to compute the face index if we are updating a face that isn't ours
          const int ghost_idx = (kernel_type == EXTERIOR_KERNEL_ALL && d != thread_dim) ?
            ghostFaceIndex<0, Arg::nDim>(coord, arg.dc.X, d, arg.nFace) :
            idx;

          const int gauge_ghost_idx = (Arg::nDim == 5 ? ghost_idx % arg.dc.ghostFaceCB[d] : ghost_idx);
          Link U = dslash_double_store() ? static_cast<const Link&>(arg.Uback(d, gauge_idx, gauge_parity)) :
                                           static_cast<const Link &>(arg.U.Ghost(d, gauge_ghost_idx, 1 - gauge_parity));
          HalfVector in = arg.halo.Ghost(d, 0, ghost_idx + (src_idx * arg.Ls + coord.s) * arg.dc.ghostFaceCB[d],
                                         their_spinor_parity);

          out += bwd_coeff * (conj(U) * in).reconstruct(d, proj_dir);
        }

        if constexpr (doBulk<kernel_type>()) {
          if constexpr (sizeof(real) == sizeof(float) && Arg::nColor == 3) {
            if ((QUDA_WILSON_MMA_DIRS == 8) || (QUDA_WILSON_MMA_DIRS == 4 && d >= 2) || (QUDA_WILSON_MMA_DIRS >= 1 && d == 3)) {
              const bool has_hop = !ghost;
              Link U_conj;
              HalfVector in_proj;
              if (has_hop) {
                Link U = dslash_double_store() ? static_cast<const Link &>(arg.Uback(d, gauge_idx, gauge_parity)) :
                                                 static_cast<const Link &>(arg.U(d, gauge_idx, 1 - gauge_parity));
                U_conj = conj(U);
                Vector in = arg.in[src_idx](back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
                in_proj = in.project(d, proj_dir);
              }
              HalfVector uv = mma_su3_mul_halfvec(U_conj, in_proj, has_hop);
              if (has_hop) {
                out += bwd_coeff * uv.reconstruct(d, proj_dir);
              }
            } else {
              if (!ghost) {
                Link U = dslash_double_store() ? static_cast<const Link &>(arg.Uback(d, gauge_idx, gauge_parity)) :
                                                 static_cast<const Link &>(arg.U(d, gauge_idx, 1 - gauge_parity));
                Vector in = arg.in[src_idx](back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
                out += bwd_coeff * (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
              }
            }
          } else {
            if (!ghost) {
              Link U = dslash_double_store() ? static_cast<const Link &>(arg.Uback(d, gauge_idx, gauge_parity)) :
                                               static_cast<const Link &>(arg.U(d, gauge_idx, 1 - gauge_parity));
              Vector in = arg.in[src_idx](back_idx + coord.s * arg.dc.volume_4d_cb, their_spinor_parity);
              out += bwd_coeff * (conj(U) * in.project(d, proj_dir)).reconstruct(d, proj_dir);
            }
          }

          prefetch(d, 1, coord, parity, arg); // prefetch the gauge link Arg::prefetch_distance ahead
        }
      }
    } // nDim
  }

  template <bool dagger, bool xpay, KernelType kernel_type, typename Arg> struct wilson : dslash_default {

    const Arg &arg;
    template <typename Ftor> constexpr wilson(const Ftor &ftor) : arg(ftor.arg) { }
    static constexpr const char *filename() { return KERNEL_FILE; } // this file name - used for run-time compilation

    // out(x) = M*in = (-D + m) * in(x-mu)
    template <KernelType mykernel_type = kernel_type>
    __device__ __host__ __forceinline__ void operator()(int idx, int src_idx, int parity)
    {
      typedef typename mapper<typename Arg::Float>::type real;
      typedef ColorSpinor<real, Arg::nColor, 4> Vector;

      bool active
        = mykernel_type == EXTERIOR_KERNEL_ALL ? false : true; // is thread active (non-trival for fused kernel only)
      int thread_dim;                                        // which dimension is thread working on (fused kernel only)

      auto coord = getCoords<QUDA_4D_PC, mykernel_type>(arg, idx, 0, parity, thread_dim);

      const int my_spinor_parity = arg.nParity == 2 ? parity : 0;
      int xs = coord.x_cb + coord.s * arg.dc.volume_4d_cb;
      Vector out;
      if (arg.dd_out.isZero(coord)) {
        if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](xs, my_spinor_parity) = out;
        return;
      }

      applyWilson<dagger, mykernel_type>(out, arg, coord, parity, idx, thread_dim, active, src_idx);

      if (xpay && mykernel_type == INTERIOR_KERNEL && arg.dd_x.isZero(coord)) {
        out = arg.a * out;
      } else if (xpay && mykernel_type == INTERIOR_KERNEL) {
        Vector x = arg.x[src_idx](xs, my_spinor_parity);\
        out = x + arg.a * out;
      } else if (mykernel_type != INTERIOR_KERNEL && active) {
        Vector x = arg.out[src_idx](xs, my_spinor_parity);
        out = x + (xpay ? arg.a * out : out);
      }

      if (mykernel_type != EXTERIOR_KERNEL_ALL || active) arg.out[src_idx](xs, my_spinor_parity) = out;
    }
  };

} // namespace quda
