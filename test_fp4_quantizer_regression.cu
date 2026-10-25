#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum ScalingScheme : int {
  SCHEME_OLD_EXP_TRUNC = 0,
  SCHEME_NAIVE_DIV = 1,
  SCHEME_FIXED_EXACT_MAX = 2,
  SCHEME_FIXED_BLOCK_UE4M3 = 3
};

__device__ __forceinline__ uint8_t cvt_f32x2_to_e2m1x2(float lo, float hi) {
  unsigned short s = 0;
  asm volatile(
      "{ .reg .b8 __t; cvt.rn.satfinite.e2m1x2.f32 __t, %2, %1; mov.b16 %0, {__t, 0}; }\n"
      : "=h"(s)
      : "f"(lo), "f"(hi));
  return static_cast<uint8_t>(s & 0xffu);
}

__device__ __forceinline__ float decode_e2m1_nibble(uint8_t nib) {
  const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float mag = lut[nib & 0x7u];
  return (nib & 0x8u) ? -mag : mag;
}

__device__ __forceinline__ float fmax_nan_f32(float a, float b) {
  float res;
  asm("max.NaN.f32 %0, %1, %2;" : "=f"(res) : "f"(a), "f"(b));
  return res;
}

__device__ __forceinline__ void compute_spinor_scales(
    const float in12[12], int scheme, float &my_vmax, float &my_scale,
    float &my_inv_scale, uint32_t &sfb) {
  sfb = 0x38383838u; // neutral 1.0 in UE4M3 (bias 7 -> 7 << 3 = 0x38)
  if (scheme == SCHEME_OLD_EXP_TRUNC) {
    my_vmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 12; i++) {
      my_vmax = fmaxf(my_vmax, fabsf(in12[i]));
    }
    my_scale = 0.0f;
    my_inv_scale = 0.0f;
    if (my_vmax > 0.0f) {
      int exp_bits = (__float_as_int(my_vmax) >> 23) & 0xff;
      my_scale = __int_as_float(exp_bits << 23) * (1.0f / 6.0f);
      my_inv_scale = __int_as_float((254 - exp_bits) << 23) * 6.0f;
    }
  } else if (scheme == SCHEME_NAIVE_DIV) {
    my_vmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 12; i++) {
      my_vmax = fmaxf(my_vmax, fabsf(in12[i]));
    }
    my_scale = 0.0f;
    my_inv_scale = 0.0f;
    // Deliberately use raw division so -ftz=true exposes 1.0f / subnormal -> +Inf -> 0 * Inf = NaN
    if (__float_as_uint(my_vmax) != 0u) {
      my_scale = my_vmax * (1.0f / 6.0f);
      my_inv_scale = 6.0f / my_vmax;
    }
  } else if (scheme == SCHEME_FIXED_EXACT_MAX) {
    my_vmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 12; i++) {
      my_vmax = fmax_nan_f32(my_vmax, fabsf(in12[i]));
    }
    int vmax_exp = (__float_as_uint(my_vmax) >> 23) & 0xff;
    my_scale = 0.0f;
    my_inv_scale = 0.0f;
    if (vmax_exp > 0 && vmax_exp < 255) {
      // Exact max scaling mapped to FP4 full-scale 6.0f with non-FTZ PTX arithmetic.
      // Using div.rn.f32 (non-FTZ) guarantees |val * my_inv_scale| <= 6.0f on all finite normal floats.
      asm volatile("mul.rn.f32 %0, %1, %2;"
                   : "=f"(my_scale)
                   : "f"(my_vmax), "f"(1.0f / 6.0f));
      asm volatile("div.rn.f32 %0, %1, %2;"
                   : "=f"(my_inv_scale)
                   : "f"(6.0f), "f"(my_vmax));
    } else if (vmax_exp == 255) {
      my_scale = __int_as_float(0x7fffffff);
      my_inv_scale = 0.0f;
    }
  } else if (scheme == SCHEME_FIXED_BLOCK_UE4M3) {
    // Ablation: encode the power-of-two part of my_vmax into sub-block 0's UE4M3 scale factor
    // and keep the mantissa adjustment in my_scale. Since all 12 real/imag components reside in
    // K=0..11 (sub-block 0 of K=64), sub-block 0 scale * my_scale_mantissa is algebraically
    // identical to SCHEME_FIXED_EXACT_MAX.
    my_vmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 12; i++) {
      my_vmax = fmax_nan_f32(my_vmax, fabsf(in12[i]));
    }
    int vmax_exp = (__float_as_uint(my_vmax) >> 23) & 0xff;
    my_scale = 0.0f;
    my_inv_scale = 0.0f;
    if (vmax_exp > 0 && vmax_exp < 255) {
      int rel_exp = vmax_exp - 127;
      int ue4m3_exp = rel_exp + 7;
      if (ue4m3_exp < 1) ue4m3_exp = 1;
      if (ue4m3_exp > 14) ue4m3_exp = 14;
      uint32_t sf0 = static_cast<uint32_t>(ue4m3_exp << 3);
      sfb = 0x38383800u | sf0;
      float block_scale = ldexpf(1.0f, ue4m3_exp - 7);
      asm volatile("div.rn.f32 %0, %1, %2;"
                   : "=f"(my_inv_scale)
                   : "f"(6.0f), "f"(my_vmax));
      my_scale = (my_vmax * (1.0f / 6.0f)) / block_scale;
    } else if (vmax_exp == 255) {
      my_scale = __int_as_float(0x7fffffff);
      my_inv_scale = 0.0f;
    }
  }
}

struct ScalarCaseResult {
  float input_val;
  float sibling_val;
  float vmax;
  float quant_in_0;
  float quant_in_1;
  float fp4_val_0;
  float fp4_val_1;
  float recon_0;
  float recon_1;
  int saturated_0;
  int rounding_bound_ok;
};

__global__ void scalar_quantizer_kernel(const float *inputs_0,
                                        const float *inputs_1, int n_cases,
                                        int scheme, ScalarCaseResult *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cases) return;

  float in12[12];
  in12[0] = inputs_0[idx];
  in12[1] = inputs_1[idx];
  for (int i = 2; i < 12; i++) in12[i] = 0.0f;

  float my_vmax = 0.0f, my_scale = 0.0f, my_inv_scale = 0.0f;
  uint32_t sfb = 0x38383838u;
  compute_spinor_scales(in12, scheme, my_vmax, my_scale, my_inv_scale, sfb);

  float q0 = in12[0] * my_inv_scale;
  float q1 = in12[1] * my_inv_scale;
  uint8_t packed = cvt_f32x2_to_e2m1x2(q0, q1);
  float fp4_0 = decode_e2m1_nibble(packed & 0xfu);
  float fp4_1 = decode_e2m1_nibble((packed >> 4) & 0xfu);

  float sfb0_scale = 1.0f;
  if (scheme == SCHEME_FIXED_BLOCK_UE4M3) {
    int ue4m3_exp = ((sfb & 0xffu) >> 3) & 0xf;
    sfb0_scale = ldexpf(1.0f, ue4m3_exp - 7);
  }
  float r0 = (fp4_0 * sfb0_scale) * my_scale;
  float r1 = (fp4_1 * sfb0_scale) * my_scale;

  ScalarCaseResult res{};
  res.input_val = in12[0];
  res.sibling_val = in12[1];
  res.vmax = my_vmax;
  res.quant_in_0 = q0;
  res.quant_in_1 = q1;
  res.fp4_val_0 = fp4_0;
  res.fp4_val_1 = fp4_1;
  res.recon_0 = r0;
  res.recon_1 = r1;
  res.saturated_0 = (isfinite(q0) && fabsf(q0) > 6.000001f) ? 1 : 0;

  // In E2M1 {0, 0.5, 1, 1.5, 2, 3, 4, 6}, the max step size in [0, 6] is 2.0 (between 4 and 6),
  // so round-to-nearest max quantization error in FP4 units is <= 1.0, i.e., <= my_vmax / 6.0f.
  if (isfinite(in12[0]) && isfinite(in12[1]) && my_vmax >= 1.1754944e-38f &&
      my_vmax < 3.4e38f) {
    float bound = my_vmax * (1.0f / 6.0f) + 1e-6f * my_vmax;
    float err0 = fabsf(r0 - in12[0]);
    float err1 = fabsf(r1 - in12[1]);
    res.rounding_bound_ok = (err0 <= bound && err1 <= bound) ? 1 : 0;
  } else {
    res.rounding_bound_ok = 1;
  }
  out[idx] = res;
}

// Kernel executing the exact sm_121a mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4
// SU(3) (12x12 real/imag) x HalfSpinor (12x8) contraction for 1 warp = 8 columns (sites).
__global__ void mma_su3_halfspinor_kernel(const float *gauge18_all,
                                          const float *spinor12_all,
                                          int num_warps, int shared_gauge,
                                          int scheme, float *out12_all,
                                          float *ref12_all,
                                          int *sat_count_all) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x & 31;
  if (warp_id >= num_warps) return;

  const float *U18 =
      gauge18_all + (shared_gauge ? 0 : warp_id) * 18; // 3x3 complex = 18 floats
  const float *S_warp = spinor12_all + warp_id * 8 * 12;

  // Build exact FP32 reference for lane < 8
  if (lane < 8) {
    const float *in_s = S_warp + lane * 12;
    float *ref_s = ref12_all + (warp_id * 8 + lane) * 12;
    for (int s = 0; s < 2; s++) {
      for (int c_out = 0; c_out < 3; c_out++) {
        double sum_re = 0.0;
        double sum_im = 0.0;
        for (int c_in = 0; c_in < 3; c_in++) {
          double u_re = U18[(c_out * 3 + c_in) * 2 + 0];
          double u_im = U18[(c_out * 3 + c_in) * 2 + 1];
          double v_re = in_s[(s * 3 + c_in) * 2 + 0];
          double v_im = in_s[(s * 3 + c_in) * 2 + 1];
          sum_re += u_re * v_re - u_im * v_im;
          sum_im += u_re * v_im + u_im * v_re;
        }
        ref_s[(s * 3 + c_out) * 2 + 0] = static_cast<float>(sum_re);
        ref_s[(s * 3 + c_out) * 2 + 1] = static_cast<float>(sum_im);
      }
    }
  }

  // Shared memory staging for A (16x64) and B (64x8)
  extern __shared__ uint8_t smem_raw[];
  int local_warp = threadIdx.x / 32;
  uint32_t *smem_a = reinterpret_cast<uint32_t *>(smem_raw) + local_warp * 256;
  uint32_t *smem_b = smem_a + 128;
  float *smem_scale = reinterpret_cast<float *>(smem_b + 64);
  uint32_t *smem_sfb = reinterpret_cast<uint32_t *>(smem_scale + 8);

  for (int i = lane; i < 128; i += 32) smem_a[i] = 0u;
  for (int i = lane; i < 64; i += 32) smem_b[i] = 0u;
  __syncwarp();

  // Pack A: 12 rows (row = s*6 + c_out*2 + ri), 12 cols (col = s*6 + c_in*2 + ri_in)
  // Row stride in smem_a is 8 uint32_t (64 nibbles = 32 bytes)
  if (lane < 12) {
    int s = lane / 6;
    int rem = lane % 6;
    int c_out = rem / 2;
    int ri = rem % 2;

    float row_vals[16];
    for (int k = 0; k < 16; k++) row_vals[k] = 0.0f;
    for (int c_in = 0; c_in < 3; c_in++) {
      float u_re = U18[(c_out * 3 + c_in) * 2 + 0];
      float u_im = U18[(c_out * 3 + c_in) * 2 + 1];
      int col_re = s * 6 + c_in * 2 + 0;
      int col_im = s * 6 + c_in * 2 + 1;
      if (ri == 0) {
        row_vals[col_re] = u_re;
        row_vals[col_im] = -u_im;
      } else {
        row_vals[col_re] = u_im;
        row_vals[col_im] = u_re;
      }
    }
    uint8_t bytes[8];
    for (int pair = 0; pair < 8; pair++) {
      bytes[pair] = cvt_f32x2_to_e2m1x2(row_vals[pair * 2 + 0] * 6.0f,
                                        row_vals[pair * 2 + 1] * 6.0f);
    }
    uint32_t w0 = 0u, w1 = 0u;
    memcpy(&w0, &bytes[0], 4);
    memcpy(&w1, &bytes[4], 4);
    smem_a[lane * 8 + 0] = w0;
    smem_a[lane * 8 + 1] = w1;
  }

  // Pack B: 8 columns (lane 0..7), each column has 12 active rows in K=0..11
  if (lane < 8) {
    const float *in_s = S_warp + lane * 12;
    float my_vmax = 0.0f, my_scale = 0.0f, my_inv_scale = 0.0f;
    uint32_t sfb = 0x38383838u;
    compute_spinor_scales(in_s, scheme, my_vmax, my_scale, my_inv_scale, sfb);
    smem_scale[lane] = my_scale;
    smem_sfb[lane] = sfb;

    int sat = 0;
    float q[16];
    for (int k = 0; k < 12; k++) {
      q[k] = in_s[k] * my_inv_scale;
      if (isfinite(q[k]) && fabsf(q[k]) > 6.000001f) sat++;
    }
    for (int k = 12; k < 16; k++) q[k] = 0.0f;
    sat_count_all[warp_id * 8 + lane] = sat;

    uint8_t bytes[8];
    for (int pair = 0; pair < 8; pair++) {
      bytes[pair] = cvt_f32x2_to_e2m1x2(q[pair * 2 + 0], q[pair * 2 + 1]);
    }
    uint32_t w0 = 0u, w1 = 0u;
    memcpy(&w0, &bytes[0], 4);
    memcpy(&w1, &bytes[4], 4);
    smem_b[lane * 8 + 0] = w0;
    smem_b[lane * 8 + 1] = w1;
  }
  __syncwarp();

  // Load fragments according to m16n8k64 lane mapping validated on sm_121a:
  // group_id = lane >> 2 (0..7), tid_in_group = lane & 3 (0..3)
  int group_id = lane >> 2;
  int tid_in_group = lane & 3;

  uint32_t a_frag[4];
  a_frag[0] = smem_a[group_id * 8 + tid_in_group];
  a_frag[1] = smem_a[(group_id + 8) * 8 + tid_in_group];
  a_frag[2] = smem_a[group_id * 8 + (tid_in_group + 4)];
  a_frag[3] = smem_a[(group_id + 8) * 8 + (tid_in_group + 4)];

  uint32_t b_frag[2];
  b_frag[0] = smem_b[group_id * 8 + tid_in_group];
  b_frag[1] = smem_b[group_id * 8 + (tid_in_group + 4)];

  uint32_t sfa = 0x38383838u;
  uint32_t sfb = smem_sfb[group_id];
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::"
      "4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
      "%10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
        "r"(b_frag[0]), "r"(b_frag[1]), "r"(sfa), "r"(sfb));

  // Unpack d[0..3]:
  // For lane = group_id * 4 + tid_in_group:
  // col0 = tid_in_group * 2 + 0, col1 = tid_in_group * 2 + 1
  // row0 = group_id (0..7), row1 = group_id + 8 (8..15; only 8..11 used)
  int col0 = tid_in_group * 2 + 0;
  int col1 = tid_in_group * 2 + 1;
  int row0 = group_id;
  int row1 = group_id + 8;

  // Gauge links were scaled by * 6.0f, so divide by 6.0f upon reconstruction
  const float gauge_recon = 1.0f / 6.0f;
  float scale_col0 = smem_scale[col0] * gauge_recon;
  float scale_col1 = smem_scale[col1] * gauge_recon;

  out12_all[(warp_id * 8 + col0) * 12 + row0] = d[0] * scale_col0;
  out12_all[(warp_id * 8 + col1) * 12 + row0] = d[1] * scale_col1;
  if (row1 < 12) {
    out12_all[(warp_id * 8 + col0) * 12 + row1] = d[2] * scale_col0;
    out12_all[(warp_id * 8 + col1) * 12 + row1] = d[3] * scale_col1;
  }
}

static void generate_random_su3(std::mt19937 &rng, float U18[18]) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  // Gram-Schmidt on 3 complex 3-vectors to build a true unitary SU(3) matrix
  double v[3][3][2];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      v[i][j][0] = dist(rng);
      v[i][j][1] = dist(rng);
    }
  auto normalize = [&](int r) {
    double n2 = 0.0;
    for (int j = 0; j < 3; j++)
      n2 += v[r][j][0] * v[r][j][0] + v[r][j][1] * v[r][j][1];
    double inv = 1.0 / std::sqrt(n2);
    for (int j = 0; j < 3; j++) {
      v[r][j][0] *= inv;
      v[r][j][1] *= inv;
    }
  };
  normalize(0);
  // Orthogonalize row 1 against row 0
  double dot_re = 0.0, dot_im = 0.0;
  for (int j = 0; j < 3; j++) {
    dot_re += v[0][j][0] * v[1][j][0] + v[0][j][1] * v[1][j][1];
    dot_im += v[0][j][0] * v[1][j][1] - v[0][j][1] * v[1][j][0];
  }
  for (int j = 0; j < 3; j++) {
    v[1][j][0] -= dot_re * v[0][j][0] - dot_im * v[0][j][1];
    v[1][j][1] -= dot_re * v[0][j][1] + dot_im * v[0][j][0];
  }
  normalize(1);
  // Row 2 = conj(row0 x row1) so det(U) = +1
  for (int j = 0; j < 3; j++) {
    int j1 = (j + 1) % 3;
    int j2 = (j + 2) % 3;
    double c_re = (v[0][j1][0] * v[1][j2][0] - v[0][j1][1] * v[1][j2][1]) -
                  (v[0][j2][0] * v[1][j1][0] - v[0][j2][1] * v[1][j1][1]);
    double c_im = (v[0][j1][0] * v[1][j2][1] + v[0][j1][1] * v[1][j2][0]) -
                  (v[0][j2][0] * v[1][j1][1] + v[0][j2][1] * v[1][j1][0]);
    v[2][j][0] = c_re;
    v[2][j][1] = -c_im;
  }
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      U18[(i * 3 + j) * 2 + 0] = static_cast<float>(v[i][j][0]);
      U18[(i * 3 + j) * 2 + 1] = static_cast<float>(v[i][j][1]);
    }
}

struct ContractionSummary {
  int total_contractions = 0;
  long long saturated_components = 0;
  double rel_l2_err = 0.0;
  double mean_rel_l2_err = 0.0;
  double max_abs_err = 0.0;
};

static ContractionSummary run_contraction_batch(
    const std::vector<float> &h_gauge, const std::vector<float> &h_spinors,
    int num_warps, int shared_gauge, int scheme) {
  int total_cols = num_warps * 8;
  float *d_gauge = nullptr, *d_spinor = nullptr, *d_out = nullptr, *d_ref = nullptr;
  int *d_sat = nullptr;
  CHECK_CUDA(cudaMalloc(&d_gauge, h_gauge.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_spinor, h_spinors.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, total_cols * 12 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_ref, total_cols * 12 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sat, total_cols * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(d_gauge, h_gauge.data(), h_gauge.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_spinor, h_spinors.data(),
                        h_spinors.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  int warps_per_block = 4;
  int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
  size_t smem_bytes = warps_per_block * 256 * sizeof(uint32_t);
  mma_su3_halfspinor_kernel<<<blocks, warps_per_block * 32, smem_bytes>>>(
      d_gauge, d_spinor, num_warps, shared_gauge, scheme, d_out, d_ref, d_sat);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> h_out(total_cols * 12);
  std::vector<float> h_ref(total_cols * 12);
  std::vector<int> h_sat(total_cols);
  CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, h_ref.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_sat.data(), d_sat, h_sat.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_gauge);
  cudaFree(d_spinor);
  cudaFree(d_out);
  cudaFree(d_ref);
  cudaFree(d_sat);

  ContractionSummary sum{};
  sum.total_contractions = total_cols;
  double diff_n2_total = 0.0;
  double ref_n2_total = 0.0;
  double sum_site_rel = 0.0;
  for (int col = 0; col < total_cols; col++) {
    sum.saturated_components += h_sat[col];
    double site_diff2 = 0.0;
    double site_ref2 = 0.0;
    for (int k = 0; k < 12; k++) {
      double o = h_out[col * 12 + k];
      double r = h_ref[col * 12 + k];
      double d = std::abs(o - r);
      if (d > sum.max_abs_err) sum.max_abs_err = d;
      site_diff2 += d * d;
      site_ref2 += r * r;
    }
    diff_n2_total += site_diff2;
    ref_n2_total += site_ref2;
    if (site_ref2 > 0.0) {
      sum_site_rel += std::sqrt(site_diff2 / site_ref2);
    }
  }
  sum.rel_l2_err = std::sqrt(diff_n2_total / ref_n2_total);
  sum.mean_rel_l2_err = sum_site_rel / total_cols;
  return sum;
}

int main(int argc, char **argv) {
  std::string ckpt_path = "work/artifacts/seed_72/ckpt_gen_00072.bin";
  std::string json_out_path = "work/fp4_quantizer_regression.json";
  bool ftz_mode = false;
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--ckpt" && i + 1 < argc) {
      ckpt_path = argv[++i];
    } else if (std::string(argv[i]) == "--json-out" && i + 1 < argc) {
      json_out_path = argv[++i];
    } else if (std::string(argv[i]) == "--ftz") {
      ftz_mode = true;
    }
  }

  // 1. Scalar & Edge-Case Battery
  std::vector<float> h_in0 = {
      0.5f,       1.0f,        2.0f,        4.0f,       1.25f,
      1.5f,       1.75f,       1.9999999f,  -1.5f,      -1.75f,
      0.0f,       -0.0f,       1e-40f,      1.4e-45f,   1e38f,
      NAN,        INFINITY,    -INFINITY};
  std::vector<float> h_in1 = {
      0.25f,      0.5f,        1.0f,        2.0f,       0.5f,
      0.75f,      1.25f,       1.5f,        0.5f,       -0.5f,
      0.0f,       0.0f,        0.0f,        0.0f,       0.5e38f,
      0.0f,       0.0f,        1.0f};
  int n_cases = static_cast<int>(h_in0.size());

  float *d_in0 = nullptr, *d_in1 = nullptr;
  ScalarCaseResult *d_res = nullptr;
  CHECK_CUDA(cudaMalloc(&d_in0, n_cases * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_in1, n_cases * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_res, n_cases * sizeof(ScalarCaseResult)));
  CHECK_CUDA(cudaMemcpy(d_in0, h_in0.data(), n_cases * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_in1, h_in1.data(), n_cases * sizeof(float),
                        cudaMemcpyHostToDevice));

  std::vector<ScalarCaseResult> res_old(n_cases), res_naive(n_cases),
      res_fixed(n_cases);

  scalar_quantizer_kernel<<<1, 32>>>(d_in0, d_in1, n_cases,
                                     SCHEME_OLD_EXP_TRUNC, d_res);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(res_old.data(), d_res,
                        n_cases * sizeof(ScalarCaseResult),
                        cudaMemcpyDeviceToHost));

  scalar_quantizer_kernel<<<1, 32>>>(d_in0, d_in1, n_cases, SCHEME_NAIVE_DIV,
                                     d_res);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(res_naive.data(), d_res,
                        n_cases * sizeof(ScalarCaseResult),
                        cudaMemcpyDeviceToHost));

  scalar_quantizer_kernel<<<1, 32>>>(d_in0, d_in1, n_cases,
                                     SCHEME_FIXED_EXACT_MAX, d_res);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(res_fixed.data(), d_res,
                        n_cases * sizeof(ScalarCaseResult),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_in0);
  cudaFree(d_in1);
  cudaFree(d_res);

  printf("=== Scalar & Edge-Case Battery (ftz=%s) ===\n",
         ftz_mode ? "true" : "false");
  int old_scalar_sat = 0;
  int fixed_scalar_sat = 0;
  int fixed_rounding_bound_all_ok = 1;
  int naive_subnormal_nan_count = 0;
  int fixed_subnormal_nan_count = 0;

  for (int i = 0; i < n_cases; i++) {
    old_scalar_sat += res_old[i].saturated_0;
    fixed_scalar_sat += res_fixed[i].saturated_0;
    if (!res_fixed[i].rounding_bound_ok) fixed_rounding_bound_all_ok = 0;
    if (i == 12 || i == 13) {
      if (std::isnan(res_naive[i].recon_0) || std::isnan(res_naive[i].recon_1))
        naive_subnormal_nan_count++;
      if (std::isnan(res_fixed[i].recon_0) || std::isnan(res_fixed[i].recon_1))
        fixed_subnormal_nan_count++;
    }
    printf(
        "Case %2d: in=(%11.4g, %10.4g) | OLD: q0=%10.4g fp4=%4.1f rec0=%11.4g "
        "| NAIVE: rec=(%10.4g,%10.4g) | FIXED: q0=%10.4g fp4=%4.1f "
        "rec=(%11.4g,%10.4g) bound_ok=%d\n",
        i, h_in0[i], h_in1[i], res_old[i].quant_in_0, res_old[i].fp4_val_0,
        res_old[i].recon_0, res_naive[i].recon_0, res_naive[i].recon_1,
        res_fixed[i].quant_in_0, res_fixed[i].fp4_val_0, res_fixed[i].recon_0,
        res_fixed[i].recon_1, res_fixed[i].rounding_bound_ok);
  }

  // Assert director oracle case (index 5: input = 1.5f)
  if (std::abs(res_old[5].quant_in_0 - 9.0f) > 1e-5f ||
      std::abs(res_old[5].recon_0 - 1.0f) > 1e-5f ||
      std::abs(res_fixed[5].quant_in_0 - 6.0f) > 1e-5f ||
      std::abs(res_fixed[5].recon_0 - 1.5f) > 1e-5f) {
    fprintf(stderr, "Assertion failed on director 1.5f oracle case!\n");
    return 2;
  }

  // Assert fixed subnormal safety (never NaN) and 1e38f exact max reconstruction
  if (fixed_subnormal_nan_count != 0) {
    fprintf(stderr, "Assertion failed: fixed scaling produced NaN on subnormals!\n");
    return 3;
  }
  if (std::abs(res_fixed[14].recon_0 - 1e38f) / 1e38f > 1e-6f) {
    fprintf(stderr, "Assertion failed on 1e38f near-FLT_MAX case: got %g\n",
            res_fixed[14].recon_0);
    return 4;
  }
  if (fixed_scalar_sat != 0 || old_scalar_sat == 0 ||
      !fixed_rounding_bound_all_ok) {
    fprintf(stderr,
            "Assertion failed: old_scalar_sat=%d fixed_scalar_sat=%d "
            "fixed_rounding_bound_all_ok=%d\n",
            old_scalar_sat, fixed_scalar_sat, fixed_rounding_bound_all_ok);
    return 5;
  }

  // 2. Non-finite lane isolation check inside an 8-column MMA tile
  std::mt19937 rng(1234567ULL);
  std::vector<float> h_u_single(18);
  generate_random_su3(rng, h_u_single.data());
  std::vector<float> h_s_nan_tile(8 * 12);
  std::normal_distribution<float> sdist(0.0f, 1.0f);
  for (int col = 0; col < 8; col++) {
    for (int k = 0; k < 12; k++) {
      h_s_nan_tile[col * 12 + k] = sdist(rng);
    }
  }
  // Inject NaN into col 3 component 0, and +Inf into col 5 component 4
  h_s_nan_tile[3 * 12 + 0] = NAN;
  h_s_nan_tile[5 * 12 + 4] = INFINITY;

  float *d_u1 = nullptr, *d_s1 = nullptr, *d_out1 = nullptr, *d_ref1 = nullptr;
  int *d_sat1 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_u1, 18 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_s1, 8 * 12 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out1, 8 * 12 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_ref1, 8 * 12 * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sat1, 8 * sizeof(int)));
  CHECK_CUDA(
      cudaMemcpy(d_u1, h_u_single.data(), 18 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_s1, h_s_nan_tile.data(), 8 * 12 * sizeof(float),
                        cudaMemcpyHostToDevice));
  mma_su3_halfspinor_kernel<<<1, 32, 256 * sizeof(uint32_t)>>>(
      d_u1, d_s1, 1, 1, SCHEME_FIXED_EXACT_MAX, d_out1, d_ref1, d_sat1);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float> h_out_nan(8 * 12);
  CHECK_CUDA(cudaMemcpy(h_out_nan.data(), d_out1, 8 * 12 * sizeof(float),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_u1);
  cudaFree(d_s1);
  cudaFree(d_out1);
  cudaFree(d_ref1);
  cudaFree(d_sat1);

  bool nan_col3_propagated = true;
  bool inf_col5_propagated = true;
  bool neighbor_lanes_finite = true;
  for (int col = 0; col < 8; col++) {
    for (int k = 0; k < 12; k++) {
      float v = h_out_nan[col * 12 + k];
      if (col == 3) {
        if (!std::isnan(v)) nan_col3_propagated = false;
      } else if (col == 5) {
        if (!std::isnan(v)) inf_col5_propagated = false;
      } else {
        if (!std::isfinite(v)) neighbor_lanes_finite = false;
      }
    }
  }
  printf(
      "Non-finite MMA tile isolation: col3_NaN_propagated=%d "
      "col5_Inf_propagated=%d neighbor_lanes_finite=%d\n",
      nan_col3_propagated ? 1 : 0, inf_col5_propagated ? 1 : 0,
      neighbor_lanes_finite ? 1 : 0);
  if (!nan_col3_propagated || !inf_col5_propagated || !neighbor_lanes_finite) {
    fprintf(stderr, "Assertion failed on non-finite MMA lane isolation!\n");
    return 6;
  }

  // 3. 65,536 Random Unitary SU(3) x HalfSpinor Contractions (8,192 warps x 8 cols)
  const int num_warps = 8192;
  const int total_contractions = num_warps * 8; // 65,536
  std::vector<float> h_rand_gauge(num_warps * 18);
  std::vector<float> h_rand_spinors(total_contractions * 12);
  std::uniform_real_distribution<float> scale_exp_dist(-6.0f, 6.0f);
  for (int w = 0; w < num_warps; w++) {
    generate_random_su3(rng, &h_rand_gauge[w * 18]);
  }
  for (int c = 0; c < total_contractions; c++) {
    float amp = std::exp2(scale_exp_dist(rng));
    for (int k = 0; k < 12; k++) {
      h_rand_spinors[c * 12 + k] = amp * sdist(rng);
    }
  }

  ContractionSummary rand_old = run_contraction_batch(
      h_rand_gauge, h_rand_spinors, num_warps, 0, SCHEME_OLD_EXP_TRUNC);
  ContractionSummary rand_fixed = run_contraction_batch(
      h_rand_gauge, h_rand_spinors, num_warps, 0, SCHEME_FIXED_EXACT_MAX);
  ContractionSummary rand_ue4m3 = run_contraction_batch(
      h_rand_gauge, h_rand_spinors, num_warps, 0, SCHEME_FIXED_BLOCK_UE4M3);

  printf(
      "Random 65,536 SU(3)xHalfSpinor:\n"
      "  OLD_EXP_TRUNC   : sat_comp=%lld / %d | rel_l2=%.6f | mean_rel_l2=%.6f | max_abs=%.6f\n"
      "  FIXED_EXACT_MAX : sat_comp=%lld / %d | rel_l2=%.6f | mean_rel_l2=%.6f | max_abs=%.6f\n"
      "  FIXED_BLK_UE4M3 : sat_comp=%lld / %d | rel_l2=%.6f | mean_rel_l2=%.6f | max_abs=%.6f\n",
      rand_old.saturated_components, total_contractions * 12,
      rand_old.rel_l2_err, rand_old.mean_rel_l2_err, rand_old.max_abs_err,
      rand_fixed.saturated_components, total_contractions * 12,
      rand_fixed.rel_l2_err, rand_fixed.mean_rel_l2_err, rand_fixed.max_abs_err,
      rand_ue4m3.saturated_components, total_contractions * 12,
      rand_ue4m3.rel_l2_err, rand_ue4m3.mean_rel_l2_err, rand_ue4m3.max_abs_err);

  // 4. Realistic SU(3) links loaded from ckpt_gen_00072.bin (8,192 links -> 65,536 contractions)
  std::ifstream ckpt_in(ckpt_path, std::ios::binary);
  if (!ckpt_in) {
    fprintf(stderr, "Failed to open checkpoint %s\n", ckpt_path.c_str());
    return 7;
  }
  std::vector<double> ckpt_links_d(num_warps * 18);
  ckpt_in.read(reinterpret_cast<char *>(ckpt_links_d.data()),
               ckpt_links_d.size() * sizeof(double));
  if (!ckpt_in) {
    fprintf(stderr, "Short read from checkpoint %s\n", ckpt_path.c_str());
    return 8;
  }
  std::vector<float> h_ckpt_gauge(num_warps * 18);
  for (size_t i = 0; i < ckpt_links_d.size(); i++) {
    h_ckpt_gauge[i] = static_cast<float>(ckpt_links_d[i]);
  }

  ContractionSummary ckpt_old = run_contraction_batch(
      h_ckpt_gauge, h_rand_spinors, num_warps, 0, SCHEME_OLD_EXP_TRUNC);
  ContractionSummary ckpt_fixed = run_contraction_batch(
      h_ckpt_gauge, h_rand_spinors, num_warps, 0, SCHEME_FIXED_EXACT_MAX);
  ContractionSummary ckpt_ue4m3 = run_contraction_batch(
      h_ckpt_gauge, h_rand_spinors, num_warps, 0, SCHEME_FIXED_BLOCK_UE4M3);

  printf(
      "Checkpoint ckpt_gen_00072.bin 65,536 SU(3)xHalfSpinor:\n"
      "  OLD_EXP_TRUNC   : sat_comp=%lld / %d | rel_l2=%.6f | mean_rel_l2=%.6f | max_abs=%.6f\n"
      "  FIXED_EXACT_MAX : sat_comp=%lld / %d | rel_l2=%.6f | mean_rel_l2=%.6f | max_abs=%.6f\n"
      "  FIXED_BLK_UE4M3 : sat_comp=%lld / %d | rel_l2=%.6f | mean_rel_l2=%.6f | max_abs=%.6f\n",
      ckpt_old.saturated_components, total_contractions * 12,
      ckpt_old.rel_l2_err, ckpt_old.mean_rel_l2_err, ckpt_old.max_abs_err,
      ckpt_fixed.saturated_components, total_contractions * 12,
      ckpt_fixed.rel_l2_err, ckpt_fixed.mean_rel_l2_err, ckpt_fixed.max_abs_err,
      ckpt_ue4m3.saturated_components, total_contractions * 12,
      ckpt_ue4m3.rel_l2_err, ckpt_ue4m3.mean_rel_l2_err, ckpt_ue4m3.max_abs_err);

  if (rand_old.saturated_components <= 0 ||
      rand_fixed.saturated_components != 0 ||
      ckpt_old.saturated_components <= 0 ||
      ckpt_fixed.saturated_components != 0 ||
      !(rand_fixed.rel_l2_err < rand_old.rel_l2_err) ||
      !(ckpt_fixed.rel_l2_err < ckpt_old.rel_l2_err)) {
    fprintf(stderr, "Assertion failed on contraction saturation or error reduction!\n");
    return 9;
  }

  std::ofstream jout(json_out_path);
  jout << "{\n"
       << "  \"ftz_mode\": " << (ftz_mode ? "true" : "false") << ",\n"
       << "  \"oracle_1p5\": {\n"
       << "    \"old_quant_in\": " << res_old[5].quant_in_0 << ",\n"
       << "    \"old_fp4_val\": " << res_old[5].fp4_val_0 << ",\n"
       << "    \"old_recon\": " << res_old[5].recon_0 << ",\n"
       << "    \"fixed_quant_in\": " << res_fixed[5].quant_in_0 << ",\n"
       << "    \"fixed_fp4_val\": " << res_fixed[5].fp4_val_0 << ",\n"
       << "    \"fixed_recon\": " << res_fixed[5].recon_0 << "\n"
       << "  },\n"
       << "  \"edge_cases\": {\n"
       << "    \"old_scalar_saturated\": " << old_scalar_sat << ",\n"
       << "    \"fixed_scalar_saturated\": " << fixed_scalar_sat << ",\n"
       << "    \"fixed_rounding_bound_all_ok\": "
       << (fixed_rounding_bound_all_ok ? "true" : "false") << ",\n"
       << "    \"naive_subnormal_nan_count\": " << naive_subnormal_nan_count
       << ",\n"
       << "    \"fixed_subnormal_nan_count\": " << fixed_subnormal_nan_count
       << ",\n"
       << "    \"near_flt_max_1e38_fixed_recon\": " << res_fixed[14].recon_0
       << ",\n"
       << "    \"nan_lane_isolated\": "
       << (nan_col3_propagated && inf_col5_propagated && neighbor_lanes_finite
               ? "true"
               : "false")
       << "\n  },\n"
       << "  \"random_su3_65536\": {\n"
       << "    \"old_saturated_components\": " << rand_old.saturated_components
       << ",\n"
       << "    \"fixed_saturated_components\": "
       << rand_fixed.saturated_components << ",\n"
       << "    \"old_rel_l2_err\": " << rand_old.rel_l2_err << ",\n"
       << "    \"fixed_rel_l2_err\": " << rand_fixed.rel_l2_err << ",\n"
       << "    \"ue4m3_block_rel_l2_err\": " << rand_ue4m3.rel_l2_err << ",\n"
       << "    \"old_mean_rel_l2_err\": " << rand_old.mean_rel_l2_err << ",\n"
       << "    \"fixed_mean_rel_l2_err\": " << rand_fixed.mean_rel_l2_err << ",\n"
       << "    \"ue4m3_block_mean_rel_l2_err\": " << rand_ue4m3.mean_rel_l2_err
       << ",\n"
       << "    \"old_max_abs_err\": " << rand_old.max_abs_err << ",\n"
       << "    \"fixed_max_abs_err\": " << rand_fixed.max_abs_err << "\n"
       << "  },\n"
       << "  \"ckpt_gen_00072_65536\": {\n"
       << "    \"old_saturated_components\": " << ckpt_old.saturated_components
       << ",\n"
       << "    \"fixed_saturated_components\": "
       << ckpt_fixed.saturated_components << ",\n"
       << "    \"old_rel_l2_err\": " << ckpt_old.rel_l2_err << ",\n"
       << "    \"fixed_rel_l2_err\": " << ckpt_fixed.rel_l2_err << ",\n"
       << "    \"ue4m3_block_rel_l2_err\": " << ckpt_ue4m3.rel_l2_err << ",\n"
       << "    \"old_mean_rel_l2_err\": " << ckpt_old.mean_rel_l2_err << ",\n"
       << "    \"fixed_mean_rel_l2_err\": " << ckpt_fixed.mean_rel_l2_err << ",\n"
       << "    \"ue4m3_block_mean_rel_l2_err\": " << ckpt_ue4m3.mean_rel_l2_err
       << ",\n"
       << "    \"old_max_abs_err\": " << ckpt_old.max_abs_err << ",\n"
       << "    \"fixed_max_abs_err\": " << ckpt_fixed.max_abs_err << "\n"
       << "  }\n"
       << "}\n";
  jout.close();
  printf("Wrote %s successfully.\n", json_out_path.c_str());
  return 0;
}
