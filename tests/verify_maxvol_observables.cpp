// Independent C++/OpenMP Observable, SU(3), Tiling-APBC & Topological Charge Verifier
// Episode: quda-max-volume-20261026 (Owner: noether)
// Verifies N_f=2 Wilson gauge configurations on 16^3x32, 24^3x48, 32^3x64, 48^3x96, 56^3x96, 56^3x112, 60^3x120, 64^3x96, 64^3x128+
// QUDA QDP/CPS Even-Odd Host Gauge Layout:
//   [mu=0..3][parity=0..1][cb_index=0..(V/2-1)][row=0..2][col=0..2][re,im] in IEEE-754 double (576 bytes/site)
//   where parity = (x + y + z + t) & 1
//         cb_index = (x / 2) + (Lx / 2) * (y + Ly * (z + Lz * t))
//   APBC convention: in hostGauge[3], links at t = Lt - 1 carry the fermion boundary sign -1
//   (so physical gauge link U_3(x, Lt-1) = -hostGauge[3](x, Lt-1) when apbc_in_file=1).

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <omp.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using Complex = std::complex<double>;

struct Mat3 {
  Complex m[3][3];
};

static inline Mat3 mat_zero() {
  Mat3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] = Complex(0.0, 0.0);
  return r;
}

static inline Mat3 mat_ident() {
  Mat3 r = mat_zero();
  r.m[0][0] = Complex(1.0, 0.0);
  r.m[1][1] = Complex(1.0, 0.0);
  r.m[2][2] = Complex(1.0, 0.0);
  return r;
}

static inline Mat3 mat_add(const Mat3 &a, const Mat3 &b) {
  Mat3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] = a.m[i][j] + b.m[i][j];
  return r;
}

static inline Mat3 mat_sub(const Mat3 &a, const Mat3 &b) {
  Mat3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] = a.m[i][j] - b.m[i][j];
  return r;
}

static inline Mat3 mat_scale(const Mat3 &a, Complex s) {
  Mat3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] = a.m[i][j] * s;
  return r;
}

static inline Mat3 mat_adj(const Mat3 &a) {
  Mat3 r;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      r.m[i][j] = std::conj(a.m[j][i]);
  return r;
}

static inline Mat3 mat_mul(const Mat3 &a, const Mat3 &b) {
  Mat3 r;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Complex s(0.0, 0.0);
      for (int k = 0; k < 3; ++k) {
        s += a.m[i][k] * b.m[k][j];
      }
      r.m[i][j] = s;
    }
  }
  return r;
}

static inline Complex mat_trace(const Mat3 &a) {
  return a.m[0][0] + a.m[1][1] + a.m[2][2];
}

static inline double mat_re_trace(const Mat3 &a) {
  return a.m[0][0].real() + a.m[1][1].real() + a.m[2][2].real();
}

static inline Complex mat_det(const Mat3 &a) {
  return a.m[0][0] * (a.m[1][1] * a.m[2][2] - a.m[1][2] * a.m[2][1]) -
         a.m[0][1] * (a.m[1][0] * a.m[2][2] - a.m[1][2] * a.m[2][0]) +
         a.m[0][2] * (a.m[1][0] * a.m[2][1] - a.m[1][1] * a.m[2][0]);
}

struct LatticeGeom {
  int L[4];
  size_t V;
  size_t Vh;
  size_t lb_last_t;

  LatticeGeom(int lx, int ly, int lz, int lt) {
    L[0] = lx;
    L[1] = ly;
    L[2] = lz;
    L[3] = lt;
    V = static_cast<size_t>(lx) * ly * lz * lt;
    Vh = V / 2;
    lb_last_t = static_cast<size_t>(lx / 2) * ly * lz * (lt - 1);
  }

  // Lexicographical index in [0, V)
  inline size_t lex_idx(int x, int y, int z, int t) const {
    return static_cast<size_t>(x) +
           static_cast<size_t>(L[0]) *
               (static_cast<size_t>(y) +
                static_cast<size_t>(L[1]) *
                    (static_cast<size_t>(z) + static_cast<size_t>(L[2]) * static_cast<size_t>(t)));
  }

  // QUDA even-odd index in [0, V)
  inline size_t eo_idx(int x, int y, int z, int t) const {
    size_t p = static_cast<size_t>((x + y + z + t) & 1);
    size_t j = static_cast<size_t>(x / 2) +
               static_cast<size_t>(L[0] / 2) *
                   (static_cast<size_t>(y) +
                    static_cast<size_t>(L[1]) *
                        (static_cast<size_t>(z) + static_cast<size_t>(L[2]) * static_cast<size_t>(t)));
    return p * Vh + j;
  }

  inline void lex_coords(size_t s, int c[4]) const {
    c[0] = static_cast<int>(s % L[0]);
    s /= L[0];
    c[1] = static_cast<int>(s % L[1]);
    s /= L[1];
    c[2] = static_cast<int>(s % L[2]);
    s /= L[2];
    c[3] = static_cast<int>(s);
  }
};

// Load physical SU(3) gauge link U_mu(x,y,z,t), stripping the baked-in APBC -1 sign at (mu==3, t==Lt-1)
static inline Mat3 load_phys_link(const double *buf, const LatticeGeom &g, int mu, int x, int y, int z, int t,
                                  bool apbc_baked) {
  size_t eo = g.eo_idx(x, y, z, t);
  const double *p = buf + ((static_cast<size_t>(mu) * g.V + eo) * 18ULL);
  double sign = (apbc_baked && mu == 3 && t == g.L[3] - 1) ? -1.0 : 1.0;
  Mat3 u;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      int idx = 2 * (3 * i + j);
      u.m[i][j] = Complex(sign * p[idx], sign * p[idx + 1]);
    }
  }
  return u;
}

static inline void shift_coords(const LatticeGeom &g, const int c[4], int dir, int step, int out[4]) {
  out[0] = c[0];
  out[1] = c[1];
  out[2] = c[2];
  out[3] = c[3];
  out[dir] = (c[dir] + step % g.L[dir] + g.L[dir]) % g.L[dir];
}

// 4-leaf clover average C_{mu,nu}(x) projected to traceless Hermitian field strength F_{mu,nu}(x)
static inline Mat3 clover_Fmunu(const double *buf, const LatticeGeom &g, const int x[4], int mu, int nu,
                                bool apbc_baked) {
  int x_pmu[4], x_pnu[4], x_mmu[4], x_mnu[4], x_mmu_pnu[4], x_mmu_mnu[4], x_pmu_mnu[4];
  shift_coords(g, x, mu, +1, x_pmu);
  shift_coords(g, x, nu, +1, x_pnu);
  shift_coords(g, x, mu, -1, x_mmu);
  shift_coords(g, x, nu, -1, x_mnu);
  shift_coords(g, x_mmu, nu, +1, x_mmu_pnu);
  shift_coords(g, x_mmu, nu, -1, x_mmu_mnu);
  shift_coords(g, x_pmu, nu, -1, x_pmu_mnu);

  auto U = [&](int d, const int c[4]) { return load_phys_link(buf, g, d, c[0], c[1], c[2], c[3], apbc_baked); };

  // Leaf 1 (top-right): U_mu(x) U_nu(x+mu) U_mu^dag(x+nu) U_nu^dag(x)
  Mat3 P = mat_mul(mat_mul(U(mu, x), U(nu, x_pmu)), mat_mul(mat_adj(U(mu, x_pnu)), mat_adj(U(nu, x))));

  // Leaf 2 (top-left): U_nu(x) U_mu^dag(x-mu+nu) U_nu^dag(x-mu) U_mu(x-mu)
  Mat3 Q = mat_mul(mat_mul(U(nu, x), mat_adj(U(mu, x_mmu_pnu))), mat_mul(mat_adj(U(nu, x_mmu)), U(mu, x_mmu)));

  // Leaf 3 (bottom-left): U_mu^dag(x-mu) U_nu^dag(x-mu-nu) U_mu(x-mu-nu) U_nu(x-nu)
  Mat3 R = mat_mul(mat_mul(mat_adj(U(mu, x_mmu)), mat_adj(U(nu, x_mmu_mnu))), mat_mul(U(mu, x_mmu_mnu), U(nu, x_mnu)));

  // Leaf 4 (bottom-right): U_nu^dag(x-nu) U_mu(x-nu) U_nu(x+mu-nu) U_mu^dag(x)
  Mat3 S = mat_mul(mat_mul(mat_adj(U(nu, x_mnu)), U(mu, x_mnu)), mat_mul(U(nu, x_pmu_mnu), mat_adj(U(mu, x))));

  Mat3 C = mat_scale(mat_add(mat_add(P, Q), mat_add(R, S)), Complex(0.25, 0.0));
  Mat3 diff = mat_sub(C, mat_adj(C));
  Mat3 F = mat_scale(diff, Complex(0.0, -0.5));
  Complex tr3 = mat_trace(F) / 3.0;
  for (int i = 0; i < 3; ++i) {
    F.m[i][i] -= tr3;
  }
  return F;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    std::fprintf(
        stderr,
        "Usage: %s <gauge.bin> <Lx> <Ly> <Lz> <Lt> [--evict-cache 0|1] [--skip-q 0|1] [--tile-k <k>]\n",
        argv[0]);
    return 2;
  }
  const char *path = argv[1];
  int Lx = std::atoi(argv[2]);
  int Ly = std::atoi(argv[3]);
  int Lz = std::atoi(argv[4]);
  int Lt = std::atoi(argv[5]);
  int evict_cache = 1;
  int skip_q = 0;
  int tile_k = 0;
  for (int i = 6; i < argc; ++i) {
    if (std::strcmp(argv[i], "--evict-cache") == 0 && i + 1 < argc) {
      evict_cache = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--skip-q") == 0 && i + 1 < argc) {
      skip_q = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--tile-k") == 0 && i + 1 < argc) {
      tile_k = std::atoi(argv[++i]);
    }
  }

  LatticeGeom g(Lx, Ly, Lz, Lt);
  size_t expected_bytes = g.V * 4ULL * 18ULL * sizeof(double);

  struct stat st;
  if (stat(path, &st) != 0) {
    std::fprintf(stderr, "ERROR: cannot stat %s\n", path);
    return 1;
  }
  if (static_cast<size_t>(st.st_size) != expected_bytes) {
    std::fprintf(stderr, "ERROR: file size mismatch for %s: got %zu, expected %zu\n", path,
                 static_cast<size_t>(st.st_size), expected_bytes);
    return 1;
  }

  double t0 = omp_get_wtime();
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    std::fprintf(stderr, "ERROR: cannot open %s\n", path);
    return 1;
  }

  size_t num_doubles = expected_bytes / sizeof(double);
  std::vector<double> buf(num_doubles);
  size_t total_read = 0;
  char *dst = reinterpret_cast<char *>(buf.data());
  while (total_read < expected_bytes) {
    ssize_t r = read(fd, dst + total_read, std::min<size_t>(expected_bytes - total_read, 1ULL << 26));
    if (r <= 0) {
      std::fprintf(stderr, "ERROR: short read on %s at %zu/%zu\n", path, total_read, expected_bytes);
      close(fd);
      return 1;
    }
    total_read += static_cast<size_t>(r);
  }
  if (evict_cache) {
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  }
  close(fd);
  double t_read = omp_get_wtime() - t0;

  // Auto-detect whether APBC -1 is baked into mu=3, t=Lt-1 by inspecting det(raw link at t=Lt-1)
  Mat3 sample_last_t = load_phys_link(buf.data(), g, 3, 0, 0, 0, Lt - 1, false);
  bool apbc_baked_in_file = (mat_det(sample_last_t).real() < 0.0);

  // 1. SU(3) unitarity, determinant, and finite-value audit across all 4V physical links
  double max_unit_err = 0.0;
  double sum_unit_err2 = 0.0;
  double max_det_err = 0.0;
  double sum_det_err2 = 0.0;
  size_t non_finite_count = 0;

  size_t total_links = 4ULL * g.V;
#pragma omp parallel for reduction(max : max_unit_err, max_det_err) \
    reduction(+ : sum_unit_err2, sum_det_err2, non_finite_count) schedule(static)
  for (size_t s = 0; s < g.V; ++s) {
    int c[4];
    g.lex_coords(s, c);
    for (int mu = 0; mu < 4; ++mu) {
      Mat3 u = load_phys_link(buf.data(), g, mu, c[0], c[1], c[2], c[3], apbc_baked_in_file);
      bool finite_ok = true;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          if (!std::isfinite(u.m[i][j].real()) || !std::isfinite(u.m[i][j].imag())) {
            finite_ok = false;
          }
        }
      }
      if (!finite_ok) {
        non_finite_count += 1;
        continue;
      }
      Mat3 udagu = mat_mul(mat_adj(u), u);
      double u_err = 0.0;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          Complex target = (i == j) ? Complex(1.0, 0.0) : Complex(0.0, 0.0);
          double d = std::abs(udagu.m[i][j] - target);
          if (d > u_err)
            u_err = d;
        }
      }
      if (u_err > max_unit_err)
        max_unit_err = u_err;
      sum_unit_err2 += u_err * u_err;

      double d_err = std::abs(mat_det(u) - Complex(1.0, 0.0));
      if (d_err > max_det_err)
        max_det_err = d_err;
      sum_det_err2 += d_err * d_err;
    }
  }

  // 2. Plaquette (spatial, temporal, total) and Clover Topological Charge Q on physical SU(3) links
  double sum_plaq_sp = 0.0;
  double sum_plaq_tp = 0.0;
  double sum_top_q = 0.0;

#pragma omp parallel for reduction(+ : sum_plaq_sp, sum_plaq_tp, sum_top_q) schedule(static)
  for (size_t s = 0; s < g.V; ++s) {
    int x[4];
    g.lex_coords(s, x);
    auto U = [&](int d, const int c[4]) {
      return load_phys_link(buf.data(), g, d, c[0], c[1], c[2], c[3], apbc_baked_in_file);
    };

    // Spatial plaquettes: (0,1), (0,2), (1,2)
    for (int mu = 0; mu < 3; ++mu) {
      for (int nu = mu + 1; nu < 3; ++nu) {
        int x_pmu[4], x_pnu[4];
        shift_coords(g, x, mu, +1, x_pmu);
        shift_coords(g, x, nu, +1, x_pnu);
        Mat3 p = mat_mul(mat_mul(U(mu, x), U(nu, x_pmu)), mat_mul(mat_adj(U(mu, x_pnu)), mat_adj(U(nu, x))));
        sum_plaq_sp += mat_re_trace(p) / 3.0;
      }
    }
    // Temporal plaquettes: (0,3), (1,3), (2,3)
    for (int mu = 0; mu < 3; ++mu) {
      int nu = 3;
      int x_pmu[4], x_pnu[4];
      shift_coords(g, x, mu, +1, x_pmu);
      shift_coords(g, x, nu, +1, x_pnu);
      Mat3 p = mat_mul(mat_mul(U(mu, x), U(nu, x_pmu)), mat_mul(mat_adj(U(mu, x_pnu)), mat_adj(U(nu, x))));
      sum_plaq_tp += mat_re_trace(p) / 3.0;
    }

    if (!skip_q) {
      Mat3 F01 = clover_Fmunu(buf.data(), g, x, 0, 1, apbc_baked_in_file);
      Mat3 F23 = clover_Fmunu(buf.data(), g, x, 2, 3, apbc_baked_in_file);
      Mat3 F02 = clover_Fmunu(buf.data(), g, x, 0, 2, apbc_baked_in_file);
      Mat3 F31 = clover_Fmunu(buf.data(), g, x, 3, 1, apbc_baked_in_file);
      Mat3 F03 = clover_Fmunu(buf.data(), g, x, 0, 3, apbc_baked_in_file);
      Mat3 F12 = clover_Fmunu(buf.data(), g, x, 1, 2, apbc_baked_in_file);

      double tr_sum = mat_re_trace(mat_mul(F01, F23)) + mat_re_trace(mat_mul(F02, F31)) +
                      mat_re_trace(mat_mul(F03, F12));
      sum_top_q += tr_sum;
    }
  }

  double plaq_spatial = sum_plaq_sp / (3.0 * static_cast<double>(g.V));
  double plaq_temporal = sum_plaq_tp / (3.0 * static_cast<double>(g.V));
  double plaq_total = 0.5 * (plaq_spatial + plaq_temporal);
  double top_q = sum_top_q / (4.0 * M_PI * M_PI);

  // 3. Polyakov loop over spatial volume V3 = Lx * Ly * Lz
  size_t V3 = static_cast<size_t>(Lx) * Ly * Lz;
  double sum_poly_phys_re = 0.0;
  double sum_poly_phys_im = 0.0;

#pragma omp parallel for reduction(+ : sum_poly_phys_re, sum_poly_phys_im) schedule(static)
  for (size_t s3 = 0; s3 < V3; ++s3) {
    int x = static_cast<int>(s3 % Lx);
    size_t rem = s3 / Lx;
    int y = static_cast<int>(rem % Ly);
    int z = static_cast<int>(rem / Ly);

    Mat3 prod = mat_ident();
    for (int t = 0; t < Lt; ++t) {
      prod = mat_mul(prod, load_phys_link(buf.data(), g, 3, x, y, z, t, apbc_baked_in_file));
    }
    Complex tr = mat_trace(prod) / 3.0;
    sum_poly_phys_re += tr.real();
    sum_poly_phys_im += tr.imag();
  }

  double poly_pbc_re = sum_poly_phys_re / static_cast<double>(V3);
  double poly_pbc_im = sum_poly_phys_im / static_cast<double>(V3);
  // QUDA gaugePolyakovLoopQuda includes the APBC -1 boundary factor on t=Lt-1:
  double poly_apbc_re = -poly_pbc_re;
  double poly_apbc_im = -poly_pbc_im;
  double poly_abs = std::hypot(poly_apbc_re, poly_apbc_im);

  // 4. Optional k^4 tiling symmetry audit (auto-check k=2 and k=3 if divisible)
  int check_k = tile_k;
  if (check_k == 0) {
    if (Lx % 3 == 0 && Ly % 3 == 0 && Lz % 3 == 0 && Lt % 3 == 0)
      check_k = 3;
    else if (Lx % 2 == 0 && Ly % 2 == 0 && Lz % 2 == 0 && Lt % 2 == 0)
      check_k = 2;
  }
  double max_tile_phys_diff = -1.0;
  if (check_k > 1 && Lx % check_k == 0 && Ly % check_k == 0 && Lz % check_k == 0 && Lt % check_k == 0) {
    int bx_len = Lx / check_k, by_len = Ly / check_k, bz_len = Lz / check_k, bt_len = Lt / check_k;
    double td = 0.0;
#pragma omp parallel for reduction(max : td) schedule(static)
    for (size_t s = 0; s < g.V; ++s) {
      int c[4];
      g.lex_coords(s, c);
      int bc[4] = {c[0] % bx_len, c[1] % by_len, c[2] % bz_len, c[3] % bt_len};
      for (int mu = 0; mu < 4; ++mu) {
        Mat3 u = load_phys_link(buf.data(), g, mu, c[0], c[1], c[2], c[3], apbc_baked_in_file);
        Mat3 ub = load_phys_link(buf.data(), g, mu, bc[0], bc[1], bc[2], bc[3], apbc_baked_in_file);
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j)
            td = std::max(td, std::abs(u.m[i][j] - ub.m[i][j]));
      }
    }
    max_tile_phys_diff = td;
  }

  double t_total = omp_get_wtime() - t0;
  bool su3_pass = (non_finite_count == 0) && (max_unit_err <= 1e-12) && (max_det_err <= 1e-12);

  std::printf("{\n");
  std::printf("  \"file\": \"%s\",\n", path);
  std::printf("  \"dims\": [%d, %d, %d, %d],\n", Lx, Ly, Lz, Lt);
  std::printf("  \"V_sites\": %zu,\n", g.V);
  std::printf("  \"file_bytes\": %zu,\n", expected_bytes);
  std::printf("  \"apbc_sign_baked_at_t_last\": %s,\n", apbc_baked_in_file ? "true" : "false");
  std::printf("  \"omp_threads\": %d,\n", omp_get_max_threads());
  std::printf("  \"read_sec\": %.4f,\n", t_read);
  std::printf("  \"total_sec\": %.4f,\n", t_total);
  std::printf("  \"su3_validation\": {\n");
  std::printf("    \"non_finite_links\": %zu,\n", non_finite_count);
  std::printf("    \"max_unitarity_err\": %.6e,\n", max_unit_err);
  std::printf("    \"rms_unitarity_err\": %.6e,\n", std::sqrt(sum_unit_err2 / static_cast<double>(total_links)));
  std::printf("    \"max_det_err\": %.6e,\n", max_det_err);
  std::printf("    \"rms_det_err\": %.6e,\n", std::sqrt(sum_det_err2 / static_cast<double>(total_links)));
  std::printf("    \"pass_1e12_gate\": %s\n", su3_pass ? "true" : "false");
  std::printf("  },\n");
  std::printf("  \"observables\": {\n");
  std::printf("    \"plaq_total\": %.15f,\n", plaq_total);
  std::printf("    \"plaq_spatial\": %.15f,\n", plaq_spatial);
  std::printf("    \"plaq_temporal\": %.15f,\n", plaq_temporal);
  std::printf("    \"polyakov_apbc_re\": %.15e,\n", poly_apbc_re);
  std::printf("    \"polyakov_apbc_im\": %.15e,\n", poly_apbc_im);
  std::printf("    \"polyakov_apbc_abs\": %.15e,\n", poly_abs);
  std::printf("    \"polyakov_pbc_re\": %.15e,\n", poly_pbc_re);
  std::printf("    \"polyakov_pbc_im\": %.15e,\n", poly_pbc_im);
  std::printf("    \"topological_charge_Q\": %.15e\n", top_q);
  std::printf("  },\n");
  std::printf("  \"tiling_apbc_diagnostics\": {\n");
  std::printf("    \"checked_tile_factor_k\": %d,\n", check_k);
  std::printf("    \"max_phys_link_tile_diff\": %.6e\n", max_tile_phys_diff);
  std::printf("  }\n");
  std::printf("}\n");

  return su3_pass ? 0 : 1;
}
