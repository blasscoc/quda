#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>
#include <string>
#include <set>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <openssl/evp.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <quda.h>
#include <gauge_field.h>
#include <hmc_quda.h>
#include "command_line_params.h"
#include "gauge_utils.h"
#include "host_utils.h"
#include "util_quda.h"

extern quda::GaugeField *gaugePrecise;

namespace quda {
  void hmcRefreshResidentGaugeState(QudaInvertParam &inv_param);
  void setHMCBaseSeed(unsigned long long seed);
  double hmcMetropolisUniform();
  void integratorResetCGStats();
  int integratorCGIters();
  int integratorCGSolves();
}

static std::string compute_sha256_buffer(const void *data, size_t len) {
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  const EVP_MD *md = EVP_sha256();
  EVP_DigestInit_ex(ctx, md, NULL);
  EVP_DigestUpdate(ctx, data, len);
  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  EVP_DigestFinal_ex(ctx, hash, &length);
  EVP_MD_CTX_free(ctx);

  std::stringstream ss;
  for (unsigned int i = 0; i < length; i++) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
  }
  return ss.str();
}

static std::string compute_sha256_file(const std::string &filepath) {
  FILE *f = fopen(filepath.c_str(), "rb");
  if (!f) return "";
  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  const EVP_MD *md = EVP_sha256();
  EVP_DigestInit_ex(ctx, md, NULL);
  unsigned char buffer[8192];
  size_t bytes = 0;
  while ((bytes = fread(buffer, 1, sizeof(buffer), f)) != 0) {
    EVP_DigestUpdate(ctx, buffer, bytes);
  }
  fclose(f);
  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int length = 0;
  EVP_DigestFinal_ex(ctx, hash, &length);
  EVP_MD_CTX_free(ctx);

  std::stringstream ss;
  for (unsigned int i = 0; i < length; i++) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
  }
  return ss.str();
}

static void get_memory_info(long &mem_available, long &vm_hwm, long &swap_used) {
  mem_available = -1;
  vm_hwm = -1;
  swap_used = -1;

  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  long swap_total = -1, swap_free = -1;
  while (std::getline(meminfo, line)) {
    if (line.find("MemAvailable:") == 0) {
      std::istringstream iss(line);
      std::string key;
      iss >> key >> mem_available; // kB
    } else if (line.find("SwapTotal:") == 0) {
      std::istringstream iss(line);
      std::string key;
      iss >> key >> swap_total;
    } else if (line.find("SwapFree:") == 0) {
      std::istringstream iss(line);
      std::string key;
      iss >> key >> swap_free;
    }
  }
  if (swap_total >= 0 && swap_free >= 0) {
    swap_used = swap_total - swap_free;
  }

  std::ifstream status("/proc/self/status");
  while (std::getline(status, line)) {
    if (line.find("VmHWM:") == 0) {
      std::istringstream iss(line);
      std::string key;
      iss >> key >> vm_hwm; // kB
    }
  }
}

static bool dir_exists(const std::string &path) {
  struct stat st;
  return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

static bool file_exists(const std::string &path) {
  struct stat st;
  return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
}

static void check_su3_host_readonly(const double *buf, const int X[4],
                                    double &max_unitarity_err, double &max_det_err,
                                    bool use_omp = false) {
  max_unitarity_err = 0.0;
  max_det_err = 0.0;
  size_t V_sites = (size_t)X[0] * (size_t)X[1] * (size_t)X[2] * (size_t)X[3];
  size_t Vh_sites = V_sites / 2;
  size_t lower_bound = ((size_t)(X[0] / 2)) * (size_t)X[1] * (size_t)X[2] * (size_t)(X[3] - 1);

  using cd = std::complex<double>;
  double u_err_max = 0.0;
  double det_err_max = 0.0;
  long nonfinite_count = 0;

  #pragma omp parallel for collapse(3) reduction(max:u_err_max,det_err_max) reduction(+:nonfinite_count) if(use_omp)
  for (int dir = 0; dir < 4; dir++) {
    for (int parity = 0; parity < 2; parity++) {
      for (size_t j = 0; j < Vh_sites; j++) {
        size_t m = (size_t)dir * V_sites + (size_t)parity * Vh_sites + j;
        const double *p = buf + m * 18;
        bool link_finite = true;
        for (int k = 0; k < 18; k++) {
          if (!std::isfinite(p[k])) {
            link_finite = false;
            break;
          }
        }
        if (!link_finite) {
          nonfinite_count += 1;
          continue;
        }
        // In QUDA QDP gauge order with QUDA_ANTI_PERIODIC_T, temporal links (dir==3)
        // on the last time slice (j >= lower_bound within each parity half) carry the
        // fermion anti-periodic boundary sign (-1), so U_raw = -U_SU3 and det(U_raw) = -1.
        // We undo this phase in temporary local storage U[3][3] without mutating buf.
        double bc_sign = (dir == 3 && j >= lower_bound) ? -1.0 : 1.0;

        cd U[3][3];
        for (int r = 0; r < 3; r++) {
          for (int c = 0; c < 3; c++) {
            int idx = 2 * (r * 3 + c);
            U[r][c] = cd(bc_sign * p[idx], bc_sign * p[idx + 1]);
          }
        }
        for (int i = 0; i < 3; i++) {
          for (int k_col = 0; k_col < 3; k_col++) {
            cd sum(0.0, 0.0);
            for (int k = 0; k < 3; k++) {
              sum += std::conj(U[k][i]) * U[k][k_col];
            }
            cd target = (i == k_col) ? cd(1.0, 0.0) : cd(0.0, 0.0);
            double err = std::abs(sum - target);
            if (!std::isfinite(err)) {
              nonfinite_count += 1;
            } else if (err > u_err_max) {
              u_err_max = err;
            }
          }
        }
        cd det = U[0][0] * (U[1][1] * U[2][2] - U[1][2] * U[2][1])
               - U[0][1] * (U[1][0] * U[2][2] - U[1][2] * U[2][0])
               + U[0][2] * (U[1][0] * U[2][1] - U[1][1] * U[2][0]);
        double derr = std::abs(det - cd(1.0, 0.0));
        if (!std::isfinite(derr)) {
          nonfinite_count += 1;
        } else if (derr > det_err_max) {
          det_err_max = derr;
        }
      }
    }
  }

  if (nonfinite_count > 0) {
    max_unitarity_err = std::numeric_limits<double>::quiet_NaN();
    max_det_err = std::numeric_limits<double>::quiet_NaN();
  } else {
    max_unitarity_err = u_err_max;
    max_det_err = det_err_max;
  }
}

static double max_link_frobenius_distance(const double *buf_a, const double *buf_b, size_t n_matrices, double &rms_dist) {
  double max_d = 0.0;
  double sum_sq = 0.0;
  for (size_t m = 0; m < n_matrices; m++) {
    const double *pa = buf_a + m * 18;
    const double *pb = buf_b + m * 18;
    double lsq = 0.0;
    for (int i = 0; i < 18; i++) {
      if (!std::isfinite(pa[i]) || !std::isfinite(pb[i])) {
        rms_dist = std::numeric_limits<double>::quiet_NaN();
        return std::numeric_limits<double>::quiet_NaN();
      }
      double d = pa[i] - pb[i];
      lsq += d * d;
    }
    double f = std::sqrt(lsq);
    if (!std::isfinite(f)) {
      rms_dist = std::numeric_limits<double>::quiet_NaN();
      return std::numeric_limits<double>::quiet_NaN();
    }
    if (f > max_d) max_d = f;
    sum_sq += lsq;
  }
  rms_dist = (n_matrices > 0) ? std::sqrt(sum_sq / (double)(n_matrices * 18)) : 0.0;
  return max_d;
}

struct TrajectorySolveAudit {
  int expected_solves = 0;
  int parsed_solves = 0;
  int quda_solves = 0;
  int parsed_iter_sum = 0;
  int quda_iter_sum = 0;
  int iter_max = 0;
  double max_iter_res = 0.0;
  double max_true_res = 0.0;
  bool has_h_init = false;
  double h_init = 0.0;
  double kin_init = 0.0;
  double gauge_init = 0.0;
  double ferm_init = 0.0;
  bool has_h_final = false;
  double h_final = 0.0;
  double kin_final = 0.0;
  double gauge_final = 0.0;
  double ferm_final = 0.0;
  double logged_dh = 0.0;
  bool exceeded_maxiter = false;
  bool healthy = false;
  std::string fail_reason;
};

static TrajectorySolveAudit parse_trajectory_quda_log(const std::string &log_text,
                                                      int expected_solves,
                                                      int quda_solves,
                                                      int quda_iters,
                                                      int maxiter,
                                                      double tol) {
  TrajectorySolveAudit aud;
  aud.expected_solves = expected_solves;
  aud.quda_solves = quda_solves;
  aud.quda_iter_sum = quda_iters;

  std::istringstream iss(log_text);
  std::string line;
  while (std::getline(iss, line)) {
    if (line.find("Exceeded maximum iterations") != std::string::npos ||
        line.find("appears to have diverged") != std::string::npos ||
        line.find("ERROR:") != std::string::npos) {
      aud.exceeded_maxiter = true;
      aud.fail_reason = "Solver warning/error detected in QUDA log: " + line;
    }

    int iters = -1;
    double iter_res = -1.0, true_res = -1.0, req_tol = -1.0;
    if (std::sscanf(line.c_str(),
                    "CG: Convergence at %d iterations, L2 relative residual: iterated = %lf, true = %lf (requested = %lf)",
                    &iters, &iter_res, &true_res, &req_tol) == 4) {
      aud.parsed_solves++;
      aud.parsed_iter_sum += iters;
      if (iters > aud.iter_max) aud.iter_max = iters;
      if (iter_res > aud.max_iter_res) aud.max_iter_res = iter_res;
      if (true_res > aud.max_true_res) aud.max_true_res = true_res;

      if (iters < 0 || iters >= maxiter || !std::isfinite(iter_res) || !std::isfinite(true_res) || true_res > tol) {
        std::ostringstream oss;
        oss << "Solve #" << aud.parsed_solves << " failed convergence check (iters=" << iters
            << "/" << maxiter << ", true_res=" << true_res << " vs tol=" << tol << ")";
        aud.fail_reason = oss.str();
      }
    }

    double h_i = 0, k_i = 0, g_i = 0, f_i = 0;
    if (std::sscanf(line.c_str(),
                    "hmcTrajectoryQuda: H_init = %lf (kinetic=%lf, gauge=%lf, fermion=%lf)",
                    &h_i, &k_i, &g_i, &f_i) == 4) {
      aud.has_h_init = true;
      aud.h_init = h_i;
      aud.kin_init = k_i;
      aud.gauge_init = g_i;
      aud.ferm_init = f_i;
    }

    double h_f = 0, k_f = 0, g_f = 0, f_f = 0, dh_v = 0;
    if (std::sscanf(line.c_str(),
                    "hmcTrajectoryQuda: H_final = %lf (kinetic=%lf, gauge=%lf, fermion=%lf), dH = %lf",
                    &h_f, &k_f, &g_f, &f_f, &dh_v) == 5) {
      aud.has_h_final = true;
      aud.h_final = h_f;
      aud.kin_final = k_f;
      aud.gauge_final = g_f;
      aud.ferm_final = f_f;
      aud.logged_dh = dh_v;
    }
  }

  if (aud.fail_reason.empty()) {
    if (aud.parsed_solves != expected_solves || aud.quda_solves != expected_solves) {
      std::ostringstream oss;
      oss << "Solve count mismatch: parsed=" << aud.parsed_solves
          << ", quda=" << aud.quda_solves << ", expected=" << expected_solves;
      aud.fail_reason = oss.str();
    } else if (aud.parsed_iter_sum != aud.quda_iter_sum) {
      std::ostringstream oss;
      oss << "Iteration sum mismatch: parsed=" << aud.parsed_iter_sum
          << " vs quda=" << aud.quda_iter_sum;
      aud.fail_reason = oss.str();
    } else if (!aud.has_h_init || !aud.has_h_final ||
               !std::isfinite(aud.h_init) || !std::isfinite(aud.h_final) ||
               !std::isfinite(aud.kin_init) || !std::isfinite(aud.gauge_init) || !std::isfinite(aud.ferm_init) ||
               !std::isfinite(aud.kin_final) || !std::isfinite(aud.gauge_final) || !std::isfinite(aud.ferm_final)) {
      aud.fail_reason = "Missing or non-finite Hamiltonian components in trajectory output";
    }
  }

  aud.healthy = aud.fail_reason.empty() && !aud.exceeded_maxiter;
  return aud;
}

static bool extract_json_number(const std::string &text, const std::string &key, double &out_val) {
  std::string pat = "\"" + key + "\"";
  size_t pos = text.find(pat);
  if (pos == std::string::npos) return false;
  pos = text.find(':', pos + pat.size());
  if (pos == std::string::npos) return false;
  pos++;
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r')) pos++;
  if (pos >= text.size()) return false;
  const char *start = text.c_str() + pos;
  char *endptr = nullptr;
  errno = 0;
  double v = std::strtod(start, &endptr);
  if (errno != 0 || endptr == start || !std::isfinite(v)) return false;
  out_val = v;
  return true;
}

static bool extract_json_string(const std::string &text, const std::string &key, std::string &out_val) {
  std::string pat = "\"" + key + "\"";
  size_t pos = text.find(pat);
  if (pos == std::string::npos) return false;
  pos = text.find(':', pos + pat.size());
  if (pos == std::string::npos) return false;
  pos = text.find('"', pos + 1);
  if (pos == std::string::npos) return false;
  size_t end = text.find('"', pos + 1);
  if (end == std::string::npos) return false;
  out_val = text.substr(pos + 1, end - (pos + 1));
  return true;
}

static bool extract_json_geometry(const std::string &text, int X[4]) {
  size_t pos = text.find("\"geometry\"");
  if (pos == std::string::npos) return false;
  size_t lb = text.find('[', pos);
  size_t rb = text.find(']', lb);
  if (lb == std::string::npos || rb == std::string::npos) return false;
  std::string arr = text.substr(lb + 1, rb - lb - 1);
  for (char &c : arr) if (c == ',') c = ' ';
  std::istringstream iss(arr);
  return (bool)(iss >> X[0] >> X[1] >> X[2] >> X[3]);
}

struct CheckpointManifest {
  int generation = 0;
  int total_traj = 0;
  int warmup_done = 0;
  int accepted_prod_traj = 0;
  int saved_confs = 0;
  int state_id = 0;
  int multiplicity = 1;
  std::string phase = "init";
  int geometry[4] = {0, 0, 0, 0};
  double beta = 0.0;
  double kappa = 0.0;
  double csw = 0.0;
  std::string integrator = "QUDA_OMELYAN_INTEGRATOR";
  double tau = 0.0;
  int n_steps = 0;
  double solver_tol = 0.0;
  int solver_maxiter = 0;
  unsigned long long seed_base = 0;
  unsigned long long seed_stride = 100000ULL;
  unsigned long long last_seed_traj = 0;
  std::string gauge_file;
  std::string gauge_sha256;
  size_t gauge_bytes = 0;
  std::string metadata_file;
  std::string metadata_sha256;
  std::string ledger_file;
  std::string ledger_sha256;
  int ledger_rows = 0;
  double plaquette_total = 0.0;
  double plaquette_spatial = 0.0;
  double plaquette_temporal = 0.0;
  double ploop_re = 0.0;
  double ploop_im = 0.0;
  double qcharge = 0.0;
  double energy_total = 0.0;
  double su3_u_err = 0.0;
  double su3_det_err = 0.0;
};

static bool load_and_validate_checkpoint_manifest(
    const std::string &out_dir,
    const std::string &manifest_path,
    int exp_X[4], double exp_beta, double exp_kappa, double exp_csw,
    double exp_tau, int exp_n_steps, double exp_tol, int exp_maxiter,
    unsigned long long exp_seed_base, unsigned long long exp_seed_stride,
    CheckpointManifest &cm,
    std::vector<char> &gauge_buf,
    std::string &err_msg) {
  std::ifstream fin(manifest_path);
  if (!fin) {
    err_msg = "Cannot open checkpoint manifest: " + manifest_path;
    return false;
  }
  std::stringstream ss;
  ss << fin.rdbuf();
  std::string content = ss.str();

  if (!extract_json_geometry(content, cm.geometry)) {
    err_msg = "Malformed checkpoint manifest: missing or invalid 'geometry'";
    return false;
  }
  for (int d = 0; d < 4; d++) {
    if (cm.geometry[d] != exp_X[d]) {
      std::ostringstream oss;
      oss << "Checkpoint geometry mismatch: manifest=[" << cm.geometry[0] << "," << cm.geometry[1]
          << "," << cm.geometry[2] << "," << cm.geometry[3] << "] vs requested=["
          << exp_X[0] << "," << exp_X[1] << "," << exp_X[2] << "," << exp_X[3] << "]";
      err_msg = oss.str();
      return false;
    }
  }

  double v = 0;
  if (!extract_json_number(content, "generation", v)) { err_msg = "Missing 'generation'"; return false; }
  cm.generation = (int)std::llround(v);
  if (!extract_json_number(content, "total_traj", v)) { err_msg = "Missing 'total_traj'"; return false; }
  cm.total_traj = (int)std::llround(v);
  if (!extract_json_number(content, "warmup_done", v)) { err_msg = "Missing 'warmup_done'"; return false; }
  cm.warmup_done = (int)std::llround(v);
  if (!extract_json_number(content, "accepted_prod_traj", v)) { err_msg = "Missing 'accepted_prod_traj'"; return false; }
  cm.accepted_prod_traj = (int)std::llround(v);
  if (!extract_json_number(content, "saved_confs", v)) { err_msg = "Missing 'saved_confs'"; return false; }
  cm.saved_confs = (int)std::llround(v);
  if (!extract_json_number(content, "state_id", v)) { err_msg = "Missing 'state_id'"; return false; }
  cm.state_id = (int)std::llround(v);
  if (!extract_json_number(content, "multiplicity", v)) { err_msg = "Missing 'multiplicity'"; return false; }
  cm.multiplicity = (int)std::llround(v);
  extract_json_string(content, "phase", cm.phase);

  if (!extract_json_number(content, "beta", cm.beta) || std::fabs(cm.beta - exp_beta) > 1e-12) {
    err_msg = "Checkpoint parameter mismatch on 'beta'";
    return false;
  }
  if (!extract_json_number(content, "kappa", cm.kappa) || std::fabs(cm.kappa - exp_kappa) > 1e-12) {
    err_msg = "Checkpoint parameter mismatch on 'kappa'";
    return false;
  }
  if (!extract_json_number(content, "csw", cm.csw) || std::fabs(cm.csw - exp_csw) > 1e-12) {
    err_msg = "Checkpoint parameter mismatch on 'csw'";
    return false;
  }
  if (!extract_json_number(content, "tau", cm.tau) || std::fabs(cm.tau - exp_tau) > 1e-12) {
    err_msg = "Checkpoint parameter mismatch on 'tau'";
    return false;
  }
  if (!extract_json_number(content, "n_steps", v) || (int)std::llround(v) != exp_n_steps) {
    err_msg = "Checkpoint parameter mismatch on 'n_steps'";
    return false;
  }
  cm.n_steps = (int)std::llround(v);
  if (!extract_json_number(content, "solver_tol", cm.solver_tol) || std::fabs(cm.solver_tol - exp_tol) > 1e-18) {
    err_msg = "Checkpoint parameter mismatch on 'solver_tol'";
    return false;
  }
  if (!extract_json_number(content, "solver_maxiter", v) || (int)std::llround(v) != exp_maxiter) {
    err_msg = "Checkpoint parameter mismatch on 'solver_maxiter'";
    return false;
  }
  cm.solver_maxiter = (int)std::llround(v);
  if (!extract_json_number(content, "seed_base", v) || (unsigned long long)std::llround(v) != exp_seed_base) {
    err_msg = "Checkpoint parameter mismatch on 'seed_base'";
    return false;
  }
  cm.seed_base = (unsigned long long)std::llround(v);
  if (!extract_json_number(content, "seed_stride", v) || (unsigned long long)std::llround(v) != exp_seed_stride) {
    err_msg = "Checkpoint parameter mismatch on 'seed_stride'";
    return false;
  }
  cm.seed_stride = (unsigned long long)std::llround(v);
  if (extract_json_number(content, "last_seed_traj", v)) cm.last_seed_traj = (unsigned long long)std::llround(v);

  if (!extract_json_string(content, "gauge_file", cm.gauge_file) ||
      !extract_json_string(content, "gauge_sha256", cm.gauge_sha256) ||
      !extract_json_number(content, "gauge_bytes", v)) {
    err_msg = "Checkpoint manifest missing gauge_file / gauge_sha256 / gauge_bytes";
    return false;
  }
  cm.gauge_bytes = (size_t)std::llround(v);
  if (!extract_json_string(content, "metadata_file", cm.metadata_file) ||
      !extract_json_string(content, "metadata_sha256", cm.metadata_sha256) ||
      !extract_json_string(content, "ledger_file", cm.ledger_file) ||
      !extract_json_string(content, "ledger_sha256", cm.ledger_sha256) ||
      !extract_json_number(content, "ledger_rows", v)) {
    err_msg = "Checkpoint manifest missing metadata/ledger fields";
    return false;
  }
  cm.ledger_rows = (int)std::llround(v);

  std::string g_ord, t_bnd, t_slc, p_cnv;
  if (!extract_json_string(content, "gauge_order", g_ord) || g_ord != "QUDA_QDP_GAUGE_ORDER") {
    err_msg = "Checkpoint manifest missing or incompatible 'gauge_order' (expected QUDA_QDP_GAUGE_ORDER)";
    return false;
  }
  if (!extract_json_string(content, "fermion_t_boundary", t_bnd) || t_bnd != "QUDA_ANTI_PERIODIC_T") {
    err_msg = "Checkpoint manifest missing or incompatible 'fermion_t_boundary' (expected QUDA_ANTI_PERIODIC_T)";
    return false;
  }
  if (!extract_json_string(content, "temporal_boundary_phase_slice", t_slc) ||
      t_slc != "dir=3,t=Lt-1,phase=-1") {
    err_msg = "Checkpoint manifest missing or incompatible 'temporal_boundary_phase_slice'";
    return false;
  }
  if (!extract_json_string(content, "ploop_convention", p_cnv) ||
      p_cnv != "QUDA_RAW_STORED_TEMPORAL_LINKS_INCLUDES_APBC_SIGN") {
    err_msg = "Checkpoint manifest missing or incompatible 'ploop_convention'";
    return false;
  }
  if (!extract_json_number(content, "plaquette_total", cm.plaquette_total) ||
      !extract_json_number(content, "plaquette_spatial", cm.plaquette_spatial) ||
      !extract_json_number(content, "plaquette_temporal", cm.plaquette_temporal) ||
      !extract_json_number(content, "ploop_re", cm.ploop_re) ||
      !extract_json_number(content, "ploop_im", cm.ploop_im) ||
      !extract_json_number(content, "qcharge", cm.qcharge) ||
      !extract_json_number(content, "energy_total", cm.energy_total) ||
      !extract_json_number(content, "su3_u_err", cm.su3_u_err) ||
      !extract_json_number(content, "su3_det_err", cm.su3_det_err)) {
    err_msg = "Checkpoint manifest missing or non-finite scientific observable metadata";
    return false;
  }

  size_t V_sites = (size_t)exp_X[0] * exp_X[1] * exp_X[2] * exp_X[3];
  size_t expected_bytes = 4 * V_sites * 18 * sizeof(double);
  if (cm.gauge_bytes != expected_bytes) {
    err_msg = "Checkpoint gauge_bytes mismatch with lattice geometry";
    return false;
  }

  std::string g_path = out_dir + "/" + cm.gauge_file;
  std::string m_path = out_dir + "/" + cm.metadata_file;
  std::string l_path = out_dir + "/" + cm.ledger_file;

  if (compute_sha256_file(m_path) != cm.metadata_sha256) {
    err_msg = "Checkpoint metadata_file SHA-256 mismatch";
    return false;
  }
  if (compute_sha256_file(l_path) != cm.ledger_sha256) {
    err_msg = "Checkpoint ledger_file SHA-256 mismatch";
    return false;
  }

  // Verify ledger row count matches cm.ledger_rows == cm.total_traj
  {
    std::ifstream fl(l_path);
    std::string line;
    int data_rows = 0;
    while (std::getline(fl, line)) {
      if (line.empty() || line.find("traj\t") == 0) continue;
      data_rows++;
    }
    if (data_rows != cm.ledger_rows || data_rows != cm.total_traj) {
      std::ostringstream oss;
      oss << "Checkpoint ledger row count mismatch: file_rows=" << data_rows
          << ", manifest_rows=" << cm.ledger_rows << ", total_traj=" << cm.total_traj;
      err_msg = oss.str();
      return false;
    }
  }

  gauge_buf.resize(expected_bytes);
  FILE *fg = fopen(g_path.c_str(), "rb");
  if (!fg) {
    err_msg = "Cannot open checkpoint gauge file: " + g_path;
    return false;
  }
  size_t rb = fread(gauge_buf.data(), 1, expected_bytes, fg);
  fclose(fg);
  if (rb != expected_bytes) {
    err_msg = "Checkpoint gauge file truncated";
    return false;
  }
  std::string g_sha = compute_sha256_buffer(gauge_buf.data(), expected_bytes);
  if (g_sha != cm.gauge_sha256) {
    err_msg = "Checkpoint gauge file SHA-256 mismatch";
    return false;
  }

  double u_err = 0.0, det_err = 0.0;
  check_su3_host_readonly(reinterpret_cast<const double *>(gauge_buf.data()), exp_X, u_err, det_err);
  if (!std::isfinite(u_err) || !std::isfinite(det_err) || u_err > 1e-12 || det_err > 1e-12) {
    err_msg = "Checkpoint gauge field failed read-only SU(3) check";
    return false;
  }

  return true;
}

static std::string parent_dir_of(const std::string &filepath) {
  size_t pos = filepath.find_last_of('/');
  if (pos == std::string::npos) return ".";
  if (pos == 0) return "/";
  return filepath.substr(0, pos);
}

static bool fsync_parent_dir(const std::string &filepath) {
  std::string pdir = parent_dir_of(filepath);
  int dfd = open(pdir.c_str(), O_RDONLY | O_DIRECTORY);
  if (dfd < 0) return false;
  int rc = fsync(dfd);
  int cl = close(dfd);
  return (rc == 0 && cl == 0);
}

static bool write_file_atomic_with_fsync(const std::string &path, const void *data, size_t len) {
  std::string tmp = path + ".tmp." + std::to_string(getpid());
  FILE *f = fopen(tmp.c_str(), "wb");
  if (!f) return false;
  if (len > 0 && fwrite(data, 1, len, f) != len) {
    fclose(f);
    unlink(tmp.c_str());
    return false;
  }
  if (fflush(f) != 0) {
    fclose(f);
    unlink(tmp.c_str());
    return false;
  }
  int fd = fileno(f);
  if (fd < 0 || fsync(fd) != 0) {
    fclose(f);
    unlink(tmp.c_str());
    return false;
  }
  if (fclose(f) != 0) {
    unlink(tmp.c_str());
    return false;
  }
  if (rename(tmp.c_str(), path.c_str()) != 0) {
    unlink(tmp.c_str());
    return false;
  }
  if (!fsync_parent_dir(path)) {
    return false;
  }
  return true;
}

static bool publish_checkpoint_generation(const std::string &out_dir,
                                          CheckpointManifest &cm,
                                          const void *gauge_data,
                                          size_t gauge_bytes,
                                          const std::vector<std::string> &ledger_lines) {
  char g_name[128], m_name[128], l_name[128];
  snprintf(g_name, sizeof(g_name), "ckpt_gen_%05d.bin", cm.generation);
  snprintf(m_name, sizeof(m_name), "ckpt_gen_%05d.meta.json", cm.generation);
  snprintf(l_name, sizeof(l_name), "ckpt_gen_%05d.ledger.tsv", cm.generation);

  cm.gauge_file = g_name;
  cm.gauge_bytes = gauge_bytes;
  if (cm.gauge_sha256.empty()) {
    cm.gauge_sha256 = compute_sha256_buffer(gauge_data, gauge_bytes);
  }
  cm.metadata_file = m_name;
  cm.ledger_file = l_name;
  cm.ledger_rows = (int)ledger_lines.size() - 1;

  std::string g_path = out_dir + "/" + g_name;
  if (!write_file_atomic_with_fsync(g_path, gauge_data, gauge_bytes)) return false;

  std::ostringstream l_ss;
  for (const auto &ln : ledger_lines) l_ss << ln << "\n";
  std::string l_str = l_ss.str();
  std::string l_path = out_dir + "/" + l_name;
  if (!write_file_atomic_with_fsync(l_path, l_str.data(), l_str.size())) return false;
  cm.ledger_sha256 = compute_sha256_buffer(l_str.data(), l_str.size());

  std::ostringstream m_ss;
  m_ss << std::setprecision(17);
  m_ss << "{\n"
       << "  \"generation\": " << cm.generation << ",\n"
       << "  \"total_traj\": " << cm.total_traj << ",\n"
       << "  \"phase\": \"" << cm.phase << "\",\n"
       << "  \"state_id\": " << cm.state_id << ",\n"
       << "  \"multiplicity\": " << cm.multiplicity << ",\n"
       << "  \"geometry\": [" << cm.geometry[0] << ", " << cm.geometry[1] << ", " << cm.geometry[2] << ", " << cm.geometry[3] << "],\n"
       << "  \"gauge_order\": \"QUDA_QDP_GAUGE_ORDER\",\n"
       << "  \"fermion_t_boundary\": \"QUDA_ANTI_PERIODIC_T\",\n"
       << "  \"temporal_boundary_phase_slice\": \"dir=3,t=Lt-1,phase=-1\",\n"
       << "  \"ploop_convention\": \"QUDA_RAW_STORED_TEMPORAL_LINKS_INCLUDES_APBC_SIGN\",\n"
       << "  \"beta\": " << cm.beta << ",\n"
       << "  \"kappa\": " << cm.kappa << ",\n"
       << "  \"csw\": " << cm.csw << ",\n"
       << "  \"integrator\": \"" << cm.integrator << "\",\n"
       << "  \"tau\": " << cm.tau << ",\n"
       << "  \"n_steps\": " << cm.n_steps << ",\n"
       << "  \"solver_tol\": " << cm.solver_tol << ",\n"
       << "  \"solver_maxiter\": " << cm.solver_maxiter << ",\n"
       << "  \"seed_base\": " << cm.seed_base << ",\n"
       << "  \"seed_stride\": " << cm.seed_stride << ",\n"
       << "  \"last_seed_traj\": " << cm.last_seed_traj << ",\n"
       << "  \"gauge_file\": \"" << cm.gauge_file << "\",\n"
       << "  \"sha256\": \"" << cm.gauge_sha256 << "\",\n"
       << "  \"bytes\": " << cm.gauge_bytes << ",\n"
       << "  \"plaquette_total\": " << cm.plaquette_total << ",\n"
       << "  \"plaquette_spatial\": " << cm.plaquette_spatial << ",\n"
       << "  \"plaquette_temporal\": " << cm.plaquette_temporal << ",\n"
       << "  \"ploop_re\": " << cm.ploop_re << ",\n"
       << "  \"ploop_im\": " << cm.ploop_im << ",\n"
       << "  \"qcharge\": " << cm.qcharge << ",\n"
       << "  \"energy_total\": " << cm.energy_total << ",\n"
       << "  \"su3_u_err\": " << cm.su3_u_err << ",\n"
       << "  \"su3_det_err\": " << cm.su3_det_err << "\n"
       << "}\n";
  std::string m_str = m_ss.str();
  std::string m_path = out_dir + "/" + m_name;
  if (!write_file_atomic_with_fsync(m_path, m_str.data(), m_str.size())) return false;
  cm.metadata_sha256 = compute_sha256_buffer(m_str.data(), m_str.size());

  // Now atomically publish checkpoint_manifest.json LAST
  std::ostringstream man_ss;
  man_ss << std::setprecision(17);
  man_ss << "{\n"
         << "  \"generation\": " << cm.generation << ",\n"
         << "  \"total_traj\": " << cm.total_traj << ",\n"
         << "  \"warmup_done\": " << cm.warmup_done << ",\n"
         << "  \"accepted_prod_traj\": " << cm.accepted_prod_traj << ",\n"
         << "  \"saved_confs\": " << cm.saved_confs << ",\n"
         << "  \"state_id\": " << cm.state_id << ",\n"
         << "  \"multiplicity\": " << cm.multiplicity << ",\n"
         << "  \"phase\": \"" << cm.phase << "\",\n"
         << "  \"geometry\": [" << cm.geometry[0] << ", " << cm.geometry[1] << ", " << cm.geometry[2] << ", " << cm.geometry[3] << "],\n"
         << "  \"gauge_order\": \"QUDA_QDP_GAUGE_ORDER\",\n"
         << "  \"fermion_t_boundary\": \"QUDA_ANTI_PERIODIC_T\",\n"
         << "  \"temporal_boundary_phase_slice\": \"dir=3,t=Lt-1,phase=-1\",\n"
         << "  \"ploop_convention\": \"QUDA_RAW_STORED_TEMPORAL_LINKS_INCLUDES_APBC_SIGN\",\n"
         << "  \"beta\": " << cm.beta << ",\n"
         << "  \"kappa\": " << cm.kappa << ",\n"
         << "  \"csw\": " << cm.csw << ",\n"
         << "  \"integrator\": \"" << cm.integrator << "\",\n"
         << "  \"tau\": " << cm.tau << ",\n"
         << "  \"n_steps\": " << cm.n_steps << ",\n"
         << "  \"solver_tol\": " << cm.solver_tol << ",\n"
         << "  \"solver_maxiter\": " << cm.solver_maxiter << ",\n"
         << "  \"seed_base\": " << cm.seed_base << ",\n"
         << "  \"seed_stride\": " << cm.seed_stride << ",\n"
         << "  \"last_seed_traj\": " << cm.last_seed_traj << ",\n"
         << "  \"gauge_file\": \"" << cm.gauge_file << "\",\n"
         << "  \"gauge_sha256\": \"" << cm.gauge_sha256 << "\",\n"
         << "  \"gauge_bytes\": " << cm.gauge_bytes << ",\n"
         << "  \"metadata_file\": \"" << cm.metadata_file << "\",\n"
         << "  \"metadata_sha256\": \"" << cm.metadata_sha256 << "\",\n"
         << "  \"ledger_file\": \"" << cm.ledger_file << "\",\n"
         << "  \"ledger_sha256\": \"" << cm.ledger_sha256 << "\",\n"
         << "  \"ledger_rows\": " << cm.ledger_rows << ",\n"
         << "  \"plaquette_total\": " << cm.plaquette_total << ",\n"
         << "  \"plaquette_spatial\": " << cm.plaquette_spatial << ",\n"
         << "  \"plaquette_temporal\": " << cm.plaquette_temporal << ",\n"
         << "  \"ploop_re\": " << cm.ploop_re << ",\n"
         << "  \"ploop_im\": " << cm.ploop_im << ",\n"
         << "  \"qcharge\": " << cm.qcharge << ",\n"
         << "  \"energy_total\": " << cm.energy_total << ",\n"
         << "  \"su3_u_err\": " << cm.su3_u_err << ",\n"
         << "  \"su3_det_err\": " << cm.su3_det_err << "\n"
         << "}\n";
  std::string man_str = man_ss.str();
  std::string man_path = out_dir + "/checkpoint_manifest.json";
  if (!write_file_atomic_with_fsync(man_path, man_str.data(), man_str.size())) return false;
  std::string main_ledger_path = out_dir + "/ensemble_ledger.tsv";
  return write_file_atomic_with_fsync(main_ledger_path, l_str.data(), l_str.size());
}

namespace quda {
  void setExtendedGaugeReconOverride(QudaReconstructType recon);
  void logResidentMemoryBreakdown(size_t V_sites, int accum_pipeline);
  bool runMultiRhsCGProbe(QudaInvertParam &inv_param, unsigned long long base_seed, size_t V_sites);
}

static void enforce_all_double_no_recon_globals(QudaPrecision sloppy_prec = QUDA_DOUBLE_PRECISION,
                                                QudaReconstructType sloppy_recon = QUDA_RECONSTRUCT_NO,
                                                QudaReconstructType precise_recon = QUDA_RECONSTRUCT_NO) {
  cpu_prec = QUDA_DOUBLE_PRECISION;
  host_gauge_data_type_size = sizeof(double);
  host_spinor_data_type_size = sizeof(double);
  host_clover_data_type_size = sizeof(double);
  prec = QUDA_DOUBLE_PRECISION;
  prec_sloppy = sloppy_prec;
  prec_precondition = sloppy_prec;
  prec_eigensolver = QUDA_DOUBLE_PRECISION;
  prec_refinement_sloppy = sloppy_prec;
  prec_null = QUDA_DOUBLE_PRECISION;
  prec_ritz = QUDA_DOUBLE_PRECISION;
  link_recon = precise_recon;
  link_recon_sloppy = sloppy_recon;
  link_recon_precondition = sloppy_recon;
  link_recon_eigensolver = precise_recon;
  setQudaPrecisions();
}

static void configure_production_params(QudaGaugeParam &gp, QudaInvertParam &ip, QudaHMCParam &hp,
                                        int lx, int ly, int lz, int lt,
                                        double beta_val, double kappa_val, double csw_val,
                                        double tau_val, int n_steps_val,
                                        int maxiter_val, double tol_val,
                                        unsigned long long seed_val,
                                        QudaPrecision sloppy_prec,
                                        QudaReconstructType sloppy_recon,
                                        double reliable_delta_val,
                                        int pipeline_val,
                                        int chrono_dim_val,
                                        int sloppy_accum_val = 0,
                                        int accum_pipeline_val = 1,
                                        QudaReconstructType precise_recon = QUDA_RECONSTRUCT_NO) {
  enforce_all_double_no_recon_globals(sloppy_prec, sloppy_recon, precise_recon);
  dim[0] = lx;
  dim[1] = ly;
  dim[2] = lz;
  dim[3] = lt;
  dslash_type = (csw_val > 0.0) ? QUDA_CLOVER_WILSON_DSLASH : QUDA_WILSON_DSLASH;
  inv_type = QUDA_CG_INVERTER;
  normalization = QUDA_KAPPA_NORMALIZATION;
  fermion_t_boundary = QUDA_ANTI_PERIODIC_T;
  anisotropy = 1.0;
  kappa = kappa_val;
  clover_csw = csw_val;
  niter = maxiter_val;
  tol = tol_val;
  reliable_delta = reliable_delta_val;
  pipeline = pipeline_val;

  gp = newQudaGaugeParam();
  setWilsonGaugeParam(gp);
  gp.X[0] = lx;
  gp.X[1] = ly;
  gp.X[2] = lz;
  gp.X[3] = lt;
  gp.type = QUDA_WILSON_LINKS;
  gp.gauge_order = QUDA_QDP_GAUGE_ORDER;
  gp.t_boundary = QUDA_ANTI_PERIODIC_T;
  gp.anisotropy = 1.0;
  gp.cpu_prec = QUDA_DOUBLE_PRECISION;
  gp.cuda_prec = QUDA_DOUBLE_PRECISION;
  gp.cuda_prec_sloppy = sloppy_prec;
  gp.cuda_prec_precondition = sloppy_prec;
  gp.cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;
  gp.cuda_prec_refinement_sloppy = sloppy_prec;
  gp.reconstruct = precise_recon;
  gp.reconstruct_sloppy = sloppy_recon;
  gp.reconstruct_precondition = sloppy_recon;
  gp.reconstruct_eigensolver = precise_recon;
  gp.reconstruct_refinement_sloppy = sloppy_recon;
  gp.use_resident_gauge = 0;
  gp.make_resident_gauge = 1;

  ip = newQudaInvertParam();
  setInvertParam(ip);
  ip.dslash_type = (csw_val > 0.0) ? QUDA_CLOVER_WILSON_DSLASH : QUDA_WILSON_DSLASH;
  ip.inv_type = QUDA_CG_INVERTER;
  ip.solve_type = QUDA_NORMOP_PC_SOLVE;
  ip.solution_type = QUDA_MATPCDAG_MATPC_SOLUTION;
  ip.matpc_type = (csw_val > 0.0) ? QUDA_MATPC_EVEN_EVEN_ASYMMETRIC : QUDA_MATPC_EVEN_EVEN;
  ip.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  ip.dagger = QUDA_DAG_NO;
  ip.cpu_prec = QUDA_DOUBLE_PRECISION;
  ip.cuda_prec = QUDA_DOUBLE_PRECISION;
  ip.cuda_prec_sloppy = sloppy_prec;
  ip.cuda_prec_precondition = sloppy_prec;
  ip.cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;
  ip.cuda_prec_refinement_sloppy = sloppy_prec;
  ip.kappa = kappa_val;
  ip.mass = 0.5 / kappa_val - 4.0;
  ip.clover_csw = csw_val;
  ip.clover_coeff = csw_val * kappa_val;
  ip.compute_true_res = 1;
  ip.verbosity = QUDA_SUMMARIZE;
  ip.Ls = 1;
  ip.maxiter = maxiter_val;
  ip.tol = tol_val;
  ip.reliable_delta = reliable_delta_val;
  ip.pipeline = pipeline_val;
  ip.use_sloppy_partial_accumulator = sloppy_accum_val;
  ip.solution_accumulator_pipeline = accum_pipeline_val;
  if (chrono_dim_val > 0) {
    ip.chrono_make_resident = 1;
    ip.chrono_use_resident = 1;
    ip.chrono_max_dim = chrono_dim_val;
    ip.chrono_index = 0;
    ip.chrono_replace_last = 0;
    ip.chrono_precision = QUDA_DOUBLE_PRECISION;
    ip.use_init_guess = QUDA_USE_INIT_GUESS_YES;
  }

  hp = newQudaHMCParam();
  hp.integrator = QUDA_OMELYAN_INTEGRATOR;
  hp.tau = tau_val;
  hp.n_steps = n_steps_val;
  hp.omelyan_lambda = 0.1931833275037836;
  hp.beta = beta_val;
  hp.generate_momentum = 1;
  hp.use_resident_gauge = 1;
  hp.make_resident_gauge = 1;
  hp.use_resident_mom = 0;
  hp.make_resident_mom = 0;
  hp.return_result_gauge = 0;
  hp.return_result_mom = 0;
  hp.momentum_seed = seed_val;
}

static std::string get_ledger_header() {
  return "traj\tphase\tstate_id\tmultiplicity\tseed_base\tseed_traj\t"
         "H_init\tH_final\tdH\texp_minus_dH\trand_val\taccepted\ttest_forced_reject\t"
         "solve_count\titer_total\titer_max\tmax_true_res\t"
         "plaq_total\tplaq_spatial\tplaq_temporal\tploop_re\tploop_im\tqcharge\tenergy_total\t"
         "su3_u_err\tsu3_det_err\ttime_sec\tvm_hwm_mb\tmem_avail_mb\tswap_used_mb\tstate_sha256";
}

int main(int argc, char **argv) {
  auto t_main_start = std::chrono::high_resolution_clock::now();
  setvbuf(stdout, NULL, _IOLBF, 0);
  setvbuf(stderr, NULL, _IOLBF, 0);
  enforce_all_double_no_recon_globals();
  niter = 2500;
  tol = 1e-10;
  dim[0] = 16;
  dim[1] = 16;
  dim[2] = 16;
  dim[3] = 32;
  hmc_beta = 5.60;
  kappa = 0.1560;
  double csw_val = 0.0;
  hmc_tau = 0.50;
  hmc_n_steps = 20;
  hmc_momentum_seed = 12345;
  unsigned long long seed_stride = 100000ULL;

  auto app = make_app();
  add_hmc_option_group(app);
  app->allow_extras();

  std::string mode = "run";
  int n_warmup = 5;
  int n_prod_target = 25;
  int max_proposals = 100;
  int save_stride = 1;
  std::string out_dir_opt = "";
  std::string test_force_reject_str = "";
  std::string sloppy_prec_str = "double";
  int sloppy_recon_int = 18;
  int precise_recon_int = 18;
  int extended_recon_int = 18;
  int omp_su3 = 0;
  int memstream_log = 0;
  int chrono_dim = 0;
  int sloppy_accum = 0;
  int accum_pipeline = 1;

  app->add_option("--mode", mode, "Execution mode: run | reversibility | validate-checkpoint | multi-rhs-probe | bench_fp4_iter");
  app->add_option("--warmup", n_warmup, "Number of warmup proposals");
  app->add_option("--prod", n_prod_target, "Target accepted production configurations");
  app->add_option("--max-proposals", max_proposals, "Maximum proposals in this invocation");
  app->add_option("--stride", save_stride, "Save stride in accepted production trajectories");
  app->add_option("--out-dir", out_dir_opt, "Output directory");
  app->add_option("--beta", hmc_beta, "Gauge coupling beta");
  app->add_option("--csw", csw_val, "Clover csw coefficient (0 for unimproved Wilson)");
  app->add_option("--tau", hmc_tau, "Trajectory length tau");
  app->add_option("--n-steps", hmc_n_steps, "Number of MD steps");
  app->add_option("--seed", hmc_momentum_seed, "Base RNG seed");
  app->add_option("--seed-stride", seed_stride, "Per-trajectory seed stride");
  app->add_option("--sloppy-prec", sloppy_prec_str, "Sloppy/precondition CG precision: double | single | half (default double)");
  app->add_option("--sloppy-recon", sloppy_recon_int, "Sloppy/precondition link reconstruction: 18 | 12 | 8 (default 18)");
  app->add_option("--precise-recon", precise_recon_int, "Authoritative gaugePrecise FP64 reconstruction: 18 | 12 | 8 (default 18)");
  app->add_option("--extended-recon", extended_recon_int, "Extended resident gauge FP64 reconstruction: 18 | 12 | 8 (default 18)");
  app->add_option("--omp-su3", omp_su3, "Enable OpenMP in read-only host SU(3) check: 0 | 1 (default 0)");
  app->add_option("--memstream-log", memstream_log, "Use open_memstream for QUDA log capture: 0 | 1 (default 0)");
  app->add_option("--chrono-dim", chrono_dim, "Per-trajectory chronological forecasting dimension (default 0)");
  app->add_option("--sloppy-accum", sloppy_accum, "Enable sloppy partial solution accumulator in CG: 0 | 1 (default 0)");
  app->add_option("--accum-pipeline", accum_pipeline, "CG solution accumulator pipeline depth Np: 1..16 (default 1)");
  app->add_option("--test-force-reject-trajs", test_force_reject_str,
                  "Comma-separated 1-based trajectory indices to force rejection (TEST-ONLY; forbidden when --prod > 0 unless --allow-test-reject)");
  bool allow_test_reject = false;
  app->add_flag("--allow-test-reject", allow_test_reject, "Explicitly mark run as a test-only forced-rejection run");

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  QudaPrecision sloppy_prec = QUDA_DOUBLE_PRECISION;
  if (sloppy_prec_str == "double") {
    sloppy_prec = QUDA_DOUBLE_PRECISION;
  } else if (sloppy_prec_str == "single") {
    sloppy_prec = QUDA_SINGLE_PRECISION;
  } else if (sloppy_prec_str == "half") {
    sloppy_prec = QUDA_HALF_PRECISION;
  } else {
    fprintf(stderr, "FATAL: Invalid --sloppy-prec '%s' (expected double | single | half)\n", sloppy_prec_str.c_str());
    return 1;
  }

  auto parse_recon_arg = [](int r_int, const char *flag_name, QudaReconstructType &out_r) -> bool {
    if (r_int == 18) { out_r = QUDA_RECONSTRUCT_NO; return true; }
    if (r_int == 12) { out_r = QUDA_RECONSTRUCT_12; return true; }
    if (r_int == 8)  { out_r = QUDA_RECONSTRUCT_8; return true; }
    fprintf(stderr, "FATAL: Invalid %s %d (expected 18 | 12 | 8)\n", flag_name, r_int);
    return false;
  };
  QudaReconstructType sloppy_recon = QUDA_RECONSTRUCT_NO;
  QudaReconstructType precise_recon = QUDA_RECONSTRUCT_NO;
  QudaReconstructType extended_recon = QUDA_RECONSTRUCT_NO;
  if (!parse_recon_arg(sloppy_recon_int, "--sloppy-recon", sloppy_recon)) return 1;
  if (!parse_recon_arg(precise_recon_int, "--precise-recon", precise_recon)) return 1;
  if (!parse_recon_arg(extended_recon_int, "--extended-recon", extended_recon)) return 1;
  if (app->count("--extended-recon") == 0 && precise_recon != QUDA_RECONSTRUCT_NO) {
    extended_recon = precise_recon;
    extended_recon_int = precise_recon_int;
  }
  quda::setExtendedGaugeReconOverride(extended_recon);

  if (app->count("--reliable-delta") == 0 && sloppy_prec != QUDA_DOUBLE_PRECISION) {
    reliable_delta = 0.01;
  }
  bool use_omp_su3 = (omp_su3 != 0);
  bool use_memstream_log = (memstream_log != 0);

  std::set<int> forced_reject_trajs;
  if (!test_force_reject_str.empty()) {
    if (!allow_test_reject) {
      fprintf(stderr, "FATAL: --test-force-reject-trajs requires --allow-test-reject and is strictly forbidden in production runs.\n");
      return 1;
    }
    std::stringstream ss(test_force_reject_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
      if (!item.empty()) forced_reject_trajs.insert(std::atoi(item.c_str()));
    }
  }

  if (out_dir_opt.empty()) {
    out_dir_opt = "work/ensemble";
  }
  mkdir(out_dir_opt.c_str(), 0777);

  // CPU-only checkpoint validation mode (runs BEFORE initQuda so malformed checkpoints fail fast without GPU use)
  if (mode == "validate-checkpoint") {
    std::string man_path = out_dir_opt + "/checkpoint_manifest.json";
    CheckpointManifest cm;
    std::vector<char> gbuf;
    std::string err_msg;
    int exp_X[4] = {dim[0], dim[1], dim[2], dim[3]};
    bool ok = load_and_validate_checkpoint_manifest(
        out_dir_opt, man_path, exp_X, hmc_beta, kappa, csw_val,
        hmc_tau, hmc_n_steps, tol, niter,
        (unsigned long long)hmc_momentum_seed, seed_stride,
        cm, gbuf, err_msg);
    if (!ok) {
      fprintf(stderr, "CHECKPOINT VALIDATION FAILED: %s\n", err_msg.c_str());
      return 1;
    }
    printf("CHECKPOINT VALIDATION PASSED: gen=%d, total_traj=%d, state_id=%d, mult=%d, sha256=%s\n",
           cm.generation, cm.total_traj, cm.state_id, cm.multiplicity, cm.gauge_sha256.c_str());
    return 0;
  }

  if (mode == "bench-fp4-iter" || mode == "bench_fp4_iter") {
    mode = "bench_fp4_iter";
    setenv("QUDA_FIXED_WORK_BENCH", "1", 0);
  }

  // If resuming from an existing checkpoint_manifest.json, validate it on CPU BEFORE initializing QUDA
  std::string man_path = out_dir_opt + "/checkpoint_manifest.json";
  CheckpointManifest cm;
  std::vector<char> resumed_gauge_buf;
  bool has_existing_checkpoint = file_exists(man_path);
  if ((mode == "run" || mode == "reversibility" || mode == "multi-rhs-probe" || mode == "bench_fp4_iter") && has_existing_checkpoint) {
    std::string err_msg;
    int exp_X[4] = {dim[0], dim[1], dim[2], dim[3]};
    if (!load_and_validate_checkpoint_manifest(
            out_dir_opt, man_path, exp_X, hmc_beta, kappa, csw_val,
            hmc_tau, hmc_n_steps, tol, niter,
            (unsigned long long)hmc_momentum_seed, seed_stride,
            cm, resumed_gauge_buf, err_msg)) {
      fprintf(stderr, "FATAL: Refusing to resume from invalid/incompatible checkpoint '%s': %s\n",
              man_path.c_str(), err_msg.c_str());
      return 1;
    }
  }

  if (!getenv("QUDA_RESOURCE_PATH")) {
    std::string tune_dir = out_dir_opt + "/../tunecache";
    mkdir(tune_dir.c_str(), 0777);
    setenv("QUDA_RESOURCE_PATH", tune_dir.c_str(), 0);
  }

  int dummy_argc = 1;
  char prog_name[] = "production_ensemble_runner";
  char *dummy_argv[] = {prog_name, nullptr};
  initComms(dummy_argc, dummy_argv, gridsize_from_cmdline);

  QudaGaugeParam gauge_param;
  QudaInvertParam inv_param;
  QudaHMCParam hmc_param;
  configure_production_params(gauge_param, inv_param, hmc_param,
                              dim[0], dim[1], dim[2], dim[3],
                              hmc_beta, kappa, csw_val, hmc_tau, hmc_n_steps,
                              niter, tol, (unsigned long long)hmc_momentum_seed,
                              sloppy_prec, sloppy_recon, reliable_delta, pipeline, chrono_dim,
                              sloppy_accum, accum_pipeline, precise_recon);

  setDims(gauge_param.X);
  initQuda(device_ordinal);
  setVerbosityQuda(QUDA_SUMMARIZE, "", stdout);
  initRand();

  size_t V_sites = (size_t)gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3];
  size_t gauge_bytes = 4 * V_sites * 18 * sizeof(double);
  std::vector<char> gauge_buf(gauge_bytes);
  std::vector<char> gauge_bkup_buf(gauge_bytes);
  void *hostGauge[4], *hostGaugeBkup[4];
  for (int d = 0; d < 4; d++) {
    hostGauge[d] = gauge_buf.data() + d * V_sites * 18 * sizeof(double);
    hostGaugeBkup[d] = gauge_bkup_buf.data() + d * V_sites * 18 * sizeof(double);
  }

  auto run_monitored_hmc_trajectory = [&](QudaHMCParam &hp_call, double &out_dH, TrajectorySolveAudit &out_audit) -> bool {
    char *mem_buf = nullptr;
    size_t mem_size = 0;
    std::string cap_path;
    FILE *cap_fp = nullptr;
    if (use_memstream_log) {
      cap_fp = open_memstream(&mem_buf, &mem_size);
      if (!cap_fp) {
        out_audit.fail_reason = "Unable to open_memstream for solver capture";
        return false;
      }
    } else {
      cap_path = out_dir_opt + "/.quda_solve_capture." + std::to_string(getpid()) + ".log";
      cap_fp = fopen(cap_path.c_str(), "w+");
      if (!cap_fp) {
        out_audit.fail_reason = "Unable to open solver capture stream: " + cap_path;
        return false;
      }
    }
    if (chrono_dim > 0) {
      flushChronoQuda(0);
    }
    setVerbosityQuda(QUDA_SUMMARIZE, "", cap_fp);
    out_dH = hmcTrajectoryQuda(nullptr, nullptr, &hp_call, &gauge_param, &inv_param, nullptr);
    fflush(cap_fp);
    setVerbosityQuda(QUDA_SUMMARIZE, "", stdout);
    if (chrono_dim > 0) {
      flushChronoQuda(0);
    }

    std::string cap_text;
    if (use_memstream_log) {
      fclose(cap_fp);
      if (mem_buf && mem_size > 0) {
        cap_text.assign(mem_buf, mem_size);
      }
      free(mem_buf);
    } else {
      fseek(cap_fp, 0, SEEK_END);
      long fsz = ftell(cap_fp);
      fseek(cap_fp, 0, SEEK_SET);
      if (fsz > 0) {
        cap_text.resize(fsz);
        fread(&cap_text[0], 1, fsz, cap_fp);
      }
      fclose(cap_fp);
      unlink(cap_path.c_str());
    }

    // Echo captured QUDA output to stdout for raw log completeness
    fwrite(cap_text.data(), 1, cap_text.size(), stdout);
    fflush(stdout);

    int expected_solves = 2 * hp_call.n_steps + 3; // Omelyan: 2*n_steps+1 force solves + 2 action solves
    if (hp_call.integrator == QUDA_LEAPFROG_INTEGRATOR) {
      expected_solves = hp_call.n_steps + 3;
    }
    int q_solves = quda::integratorCGSolves();
    int q_iters = quda::integratorCGIters();

    out_audit = parse_trajectory_quda_log(cap_text, expected_solves, q_solves, q_iters, inv_param.maxiter, inv_param.tol);
    if (!std::isfinite(out_dH)) {
      out_audit.healthy = false;
      out_audit.fail_reason = "hmcTrajectoryQuda returned non-finite dH";
    }
    return out_audit.healthy;
  };

  // Mode: Same-Gauge Multi-RHS CG Amortization Probe (Axis 2) & Fixed-Work Iteration Benchmark
  if (mode == "multi-rhs-probe" || mode == "bench_fp4_iter") {
    if (has_existing_checkpoint) {
      memcpy(gauge_buf.data(), resumed_gauge_buf.data(), gauge_bytes);
    } else {
      constructHostGaugeField(hostGauge, gauge_param, dummy_argc, dummy_argv);
    }
    loadGaugeQuda(hostGauge, &gauge_param);
    quda::hmcRefreshResidentGaugeState(inv_param);
    bool ok = quda::runMultiRhsCGProbe(inv_param, (unsigned long long)hmc_momentum_seed, V_sites);
    quda::logResidentMemoryBreakdown(V_sites, accum_pipeline);
    destroyHMCQuda();
    freeGaugeQuda();
    freeCloverQuda();
    endQuda();
    finalizeComms();
    return ok ? 0 : 1;
  }

  // Mode: Forward-Backward Reversibility Verification
  if (mode == "reversibility") {
    printf("=== Executing Forward/Backward MD Reversibility Test (%dx%dx%dx%d, beta=%.4f, kappa=%.4f, tau=%.4f, n_steps=%d, sloppy_prec=%s, sloppy_recon=%d, reliable_delta=%.4g, sloppy_accum=%d, accum_pipeline=%d, omp_su3=%d, memstream_log=%d) ===\n",
           dim[0], dim[1], dim[2], dim[3], hmc_beta, kappa, hmc_tau, hmc_n_steps,
           sloppy_prec_str.c_str(), sloppy_recon_int, reliable_delta, sloppy_accum, accum_pipeline, omp_su3, memstream_log);
    double u_err_init_host = 0.0, det_err_init_host = 0.0;
    double u_err_after_load = 0.0, det_err_after_load = 0.0;
    unsigned long long fwd_seed = (unsigned long long)hmc_momentum_seed + 2ULL * seed_stride;

    if (has_existing_checkpoint) {
      memcpy(gauge_buf.data(), resumed_gauge_buf.data(), gauge_bytes);
      check_su3_host_readonly(reinterpret_cast<const double *>(gauge_buf.data()), dim.data(), u_err_init_host, det_err_init_host, use_omp_su3);
      loadGaugeQuda(hostGauge, &gauge_param);
      quda::hmcRefreshResidentGaugeState(inv_param);
      saveGaugeQuda(hostGaugeBkup, &gauge_param);
      check_su3_host_readonly(reinterpret_cast<const double *>(gauge_bkup_buf.data()), dim.data(), u_err_after_load, det_err_after_load, use_omp_su3);
      fwd_seed = (unsigned long long)hmc_momentum_seed + (unsigned long long)(cm.total_traj + 1) * seed_stride;
      printf("Reversibility starting from checkpoint gen=%d (total_traj=%d, state_id=%d, sha256=%s, fwd_seed=%llu)\n",
             cm.generation, cm.total_traj, cm.state_id, cm.gauge_sha256.substr(0, 12).c_str(), fwd_seed);
    } else {
      constructHostGaugeField(hostGauge, gauge_param, dummy_argc, dummy_argv);
      check_su3_host_readonly(reinterpret_cast<const double *>(gauge_buf.data()), dim.data(), u_err_init_host, det_err_init_host, use_omp_su3);

      loadGaugeQuda(hostGauge, &gauge_param);
      quda::hmcRefreshResidentGaugeState(inv_param);
      saveGaugeQuda(hostGaugeBkup, &gauge_param);
      check_su3_host_readonly(reinterpret_cast<const double *>(gauge_bkup_buf.data()), dim.data(), u_err_after_load, det_err_after_load, use_omp_su3);

      // First evolve 1 short trajectory so the gauge field is non-trivial and generic (not unit/cold links)
      unsigned long long prep_seed = (unsigned long long)hmc_momentum_seed + seed_stride;
      hmc_param.momentum_seed = prep_seed;
      hmc_param.tau = hmc_tau;
      hmc_param.n_steps = hmc_n_steps;
      hmc_param.generate_momentum = 1;
      hmc_param.use_resident_mom = 0;
      hmc_param.make_resident_mom = 1;
      hmc_param.reuse_pseudofermion = 0;
      hmc_param.use_resident_gauge = 1;
      hmc_param.make_resident_gauge = 1;
      quda::setHMCBaseSeed(prep_seed);

      double prep_dH = 0.0;
      TrajectorySolveAudit prep_aud;
      if (!run_monitored_hmc_trajectory(hmc_param, prep_dH, prep_aud)) {
        fprintf(stderr, "FATAL: Prep trajectory failed solve audit: %s\n", prep_aud.fail_reason.c_str());
        destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
        return 1;
      }
    }

    // Record starting state U_start
    saveGaugeQuda(hostGaugeBkup, &gauge_param);
    double plaq_start[3];
    plaqQuda(plaq_start);
    double u_err_start = 0.0, det_err_start = 0.0;
    check_su3_host_readonly(reinterpret_cast<const double *>(gauge_bkup_buf.data()), dim.data(), u_err_start, det_err_start, use_omp_su3);
    printf("Stage SU(3) errors: init_host=(%.3e, %.3e), after_load=(%.3e, %.3e), after_prep=(%.3e, %.3e)\n",
           u_err_init_host, det_err_init_host, u_err_after_load, det_err_after_load, u_err_start, det_err_start);

    // Forward leg (+tau): generate fresh momentum and pseudofermion, keep momResident and storedPhi
    hmc_param.momentum_seed = fwd_seed;
    hmc_param.tau = hmc_tau;
    hmc_param.n_steps = hmc_n_steps;
    hmc_param.generate_momentum = 1;
    hmc_param.use_resident_mom = 0;
    hmc_param.make_resident_mom = 1;
    hmc_param.reuse_pseudofermion = 0;
    hmc_param.use_resident_gauge = 1;
    hmc_param.make_resident_gauge = 1;
    quda::setHMCBaseSeed(fwd_seed);

    double dH_fwd = 0.0;
    TrajectorySolveAudit aud_fwd;
    if (!run_monitored_hmc_trajectory(hmc_param, dH_fwd, aud_fwd)) {
      fprintf(stderr, "FATAL: Forward leg failed solve audit: %s\n", aud_fwd.fail_reason.c_str());
      destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
      return 1;
    }
    saveGaugeQuda(hostGauge, &gauge_param);
    double plaq_mid[3];
    plaqQuda(plaq_mid);
    double u_err_mid = 0.0, det_err_mid = 0.0;
    check_su3_host_readonly(reinterpret_cast<const double *>(gauge_buf.data()), dim.data(), u_err_mid, det_err_mid, use_omp_su3);

    // Backward leg (-tau): retain momResident and storedPhi without duplicate momentum negation
    hmc_param.tau = -hmc_tau;
    hmc_param.n_steps = hmc_n_steps;
    hmc_param.generate_momentum = 0;
    hmc_param.use_resident_mom = 1;
    hmc_param.make_resident_mom = 1;
    hmc_param.reuse_pseudofermion = 1;
    hmc_param.use_resident_gauge = 1;
    hmc_param.make_resident_gauge = 1;

    double dH_bwd = 0.0;
    TrajectorySolveAudit aud_bwd;
    if (!run_monitored_hmc_trajectory(hmc_param, dH_bwd, aud_bwd)) {
      fprintf(stderr, "FATAL: Backward leg failed solve audit: %s\n", aud_bwd.fail_reason.c_str());
      destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
      return 1;
    }

    saveGaugeQuda(hostGauge, &gauge_param);
    double plaq_return[3];
    plaqQuda(plaq_return);

    double u_err_ret = 0.0, det_err_ret = 0.0;
    check_su3_host_readonly(reinterpret_cast<const double *>(gauge_buf.data()), dim.data(), u_err_ret, det_err_ret, use_omp_su3);

    double rms_link_dist = 0.0;
    double max_link_dist = max_link_frobenius_distance(
        reinterpret_cast<const double *>(gauge_bkup_buf.data()),
        reinterpret_cast<const double *>(gauge_buf.data()),
        4 * V_sites, rms_link_dist);

    double dH_net = dH_fwd + dH_bwd;
    double H_return_minus_H_start = aud_bwd.h_final - aud_fwd.h_init;
    double abs_dH_rev = std::max(std::fabs(dH_net), std::fabs(H_return_minus_H_start));
    double abs_dPlaq_rev = std::fabs(plaq_return[0] - plaq_start[0]);

    std::string rev_json_path = out_dir_opt + "/reversibility_result.json";
    FILE *fr = fopen(rev_json_path.c_str(), "w");
    bool pass = std::isfinite(abs_dH_rev) && std::isfinite(abs_dPlaq_rev) &&
                std::isfinite(max_link_dist) && std::isfinite(rms_link_dist) &&
                std::isfinite(u_err_ret) && std::isfinite(det_err_ret) &&
                (abs_dH_rev < 1e-7) && (abs_dPlaq_rev < 1e-12) &&
                (u_err_ret <= 1e-12) && (det_err_ret <= 1e-12) &&
                aud_fwd.healthy && aud_bwd.healthy;
    if (fr) {
      fprintf(fr, "{\n");
      fprintf(fr, "  \"pass\": %s,\n", pass ? "true" : "false");
      fprintf(fr, "  \"geometry\": [%d, %d, %d, %d],\n", dim[0], dim[1], dim[2], dim[3]);
      fprintf(fr, "  \"beta\": %.6f,\n", hmc_beta);
      fprintf(fr, "  \"kappa\": %.6f,\n", kappa);
      fprintf(fr, "  \"tau\": %.6f,\n", hmc_tau);
      fprintf(fr, "  \"n_steps\": %d,\n", hmc_n_steps);
      fprintf(fr, "  \"sloppy_prec\": \"%s\",\n", sloppy_prec_str.c_str());
      fprintf(fr, "  \"sloppy_recon\": %d,\n", sloppy_recon_int);
      fprintf(fr, "  \"reliable_delta\": %.6g,\n", reliable_delta);
      fprintf(fr, "  \"H_start\": %.17g,\n", aud_fwd.h_init);
      fprintf(fr, "  \"H_mid\": %.17g,\n", aud_fwd.h_final);
      fprintf(fr, "  \"H_return\": %.17g,\n", aud_bwd.h_final);
      fprintf(fr, "  \"dH_fwd\": %.12e,\n", dH_fwd);
      fprintf(fr, "  \"dH_bwd\": %.12e,\n", dH_bwd);
      fprintf(fr, "  \"abs_H_return_minus_H_start\": %.12e,\n", abs_dH_rev);
      fprintf(fr, "  \"plaq_start\": %.17g,\n", plaq_start[0]);
      fprintf(fr, "  \"plaq_mid\": %.17g,\n", plaq_mid[0]);
      fprintf(fr, "  \"plaq_return\": %.17g,\n", plaq_return[0]);
      fprintf(fr, "  \"abs_plaq_diff\": %.12e,\n", abs_dPlaq_rev);
      fprintf(fr, "  \"max_link_frobenius_dist\": %.12e,\n", max_link_dist);
      fprintf(fr, "  \"rms_link_dist\": %.12e,\n", rms_link_dist);
      fprintf(fr, "  \"su3_u_err_init_host\": %.12e,\n", u_err_init_host);
      fprintf(fr, "  \"su3_det_err_init_host\": %.12e,\n", det_err_init_host);
      fprintf(fr, "  \"su3_u_err_after_load\": %.12e,\n", u_err_after_load);
      fprintf(fr, "  \"su3_det_err_after_load\": %.12e,\n", det_err_after_load);
      fprintf(fr, "  \"su3_u_err_start\": %.12e,\n", u_err_start);
      fprintf(fr, "  \"su3_det_err_start\": %.12e,\n", det_err_start);
      fprintf(fr, "  \"su3_u_err_mid\": %.12e,\n", u_err_mid);
      fprintf(fr, "  \"su3_det_err_mid\": %.12e,\n", det_err_mid);
      fprintf(fr, "  \"su3_u_err_return\": %.12e,\n", u_err_ret);
      fprintf(fr, "  \"su3_det_err_return\": %.12e,\n", det_err_ret);
      fprintf(fr, "  \"fwd_solves\": %d,\n", aud_fwd.parsed_solves);
      fprintf(fr, "  \"fwd_iter_total\": %d,\n", aud_fwd.parsed_iter_sum);
      fprintf(fr, "  \"fwd_iter_max\": %d,\n", aud_fwd.iter_max);
      fprintf(fr, "  \"fwd_max_true_res\": %.12e,\n", aud_fwd.max_true_res);
      fprintf(fr, "  \"bwd_solves\": %d,\n", aud_bwd.parsed_solves);
      fprintf(fr, "  \"bwd_iter_total\": %d,\n", aud_bwd.parsed_iter_sum);
      fprintf(fr, "  \"bwd_iter_max\": %d,\n", aud_bwd.iter_max);
      fprintf(fr, "  \"bwd_max_true_res\": %.12e\n", aud_bwd.max_true_res);
      fprintf(fr, "}\n");
      fclose(fr);
    }

    printf("Reversibility Result: |H_return - H_start| = %.6e (< 1e-7), |dPlaq| = %.6e (< 1e-12), max_link_dist = %.6e, rms_link_dist = %.6e, u_err = %.3e, det_err = %.3e, fwd_solves = %d (max_res=%.3e), bwd_solves = %d (max_res=%.3e) -> %s\n",
           abs_dH_rev, abs_dPlaq_rev, max_link_dist, rms_link_dist, u_err_ret, det_err_ret,
           aud_fwd.parsed_solves, aud_fwd.max_true_res, aud_bwd.parsed_solves, aud_bwd.max_true_res,
           pass ? "PASS" : "FAIL");

    destroyHMCQuda();
    freeGaugeQuda();
    endQuda();
    finalizeComms();
    return pass ? 0 : 1;
  }

  // Standard / Resumable Chain Execution (mode == "run")
  std::vector<std::string> ledger_lines;
  int generation = 0;
  int total_traj = 0;
  int warmup_done = 0;
  int accepted_prod_traj = 0;
  int saved_confs = 0;
  int state_id = 0;
  int multiplicity = 1;

  if (has_existing_checkpoint) {
    generation = cm.generation;
    total_traj = cm.total_traj;
    warmup_done = cm.warmup_done;
    accepted_prod_traj = cm.accepted_prod_traj;
    saved_confs = cm.saved_confs;
    state_id = cm.state_id;
    multiplicity = cm.multiplicity;
    memcpy(gauge_buf.data(), resumed_gauge_buf.data(), gauge_bytes);

    std::ifstream fl(out_dir_opt + "/" + cm.ledger_file);
    std::string ln;
    while (std::getline(fl, ln)) {
      if (!ln.empty()) ledger_lines.push_back(ln);
    }
    printf("Resumed verified checkpoint generation %d (total_traj=%d, warmup_done=%d, accepted_prod_traj=%d, saved_confs=%d, state_id=%d, mult=%d, sha256=%s)\n",
           generation, total_traj, warmup_done, accepted_prod_traj, saved_confs, state_id, multiplicity,
           cm.gauge_sha256.substr(0, 12).c_str());
  } else {
    constructHostGaugeField(hostGauge, gauge_param, dummy_argc, dummy_argv);
    ledger_lines.push_back(get_ledger_header());
  }

  loadGaugeQuda(hostGauge, &gauge_param);
  quda::hmcRefreshResidentGaugeState(inv_param);
  saveGaugeQuda(hostGauge, &gauge_param);

  double cur_u_err = 0.0, cur_det_err = 0.0;
  check_su3_host_readonly(reinterpret_cast<const double *>(gauge_buf.data()), dim.data(), cur_u_err, cur_det_err, use_omp_su3);
  if (!std::isfinite(cur_u_err) || !std::isfinite(cur_det_err) || cur_u_err > 1e-12 || cur_det_err > 1e-12) {
    fprintf(stderr, "FATAL: Initial/resumed gauge field failed read-only SU(3) check (u_err=%.3e, det_err=%.3e)\n",
            cur_u_err, cur_det_err);
    destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
    return 1;
  }

  std::string cur_sha256 = compute_sha256_buffer(gauge_buf.data(), gauge_bytes);
  double cur_plaq[3];
  plaqQuda(cur_plaq);
  if (has_existing_checkpoint && std::fabs(cur_plaq[0] - cm.plaquette_total) > 1e-12) {
    fprintf(stderr, "FATAL: Resumed device plaquette %.17g differs from checkpoint manifest %.17g by %.3e\n",
            cur_plaq[0], cm.plaquette_total, std::fabs(cur_plaq[0] - cm.plaquette_total));
    destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
    return 1;
  }

  printf("Loaded gauge state_id=%d (mult=%d): plaq_total=%.17g, su3_u_err=%.3e, su3_det_err=%.3e, sha256=%s\n",
         state_id, multiplicity, cur_plaq[0], cur_u_err, cur_det_err, cur_sha256.substr(0, 12).c_str());

  auto t_setup_done = std::chrono::high_resolution_clock::now();
  double setup_init_sec = std::chrono::duration<double>(t_setup_done - t_main_start).count();
  printf("Setup/Init Complete: setup_init_sec=%.6f, sloppy_prec=%s, sloppy_recon=%d, reliable_delta=%.4g, pipeline=%d, sloppy_accum=%d, accum_pipeline=%d, omp_su3=%d, memstream_log=%d, chrono_dim=%d\n",
         setup_init_sec, sloppy_prec_str.c_str(), sloppy_recon_int, reliable_delta, pipeline, sloppy_accum, accum_pipeline, omp_su3, memstream_log, chrono_dim);

  int proposals_this_run = 0;
  int accepted_this_run = 0;
  double sum_traj_pre_io_sec = 0.0;
  double sum_traj_complete_sec = 0.0;
  long long sum_cg_iters = 0;
  double run_max_true_res = 0.0;
  double run_max_u_err = cur_u_err;
  double run_max_det_err = cur_det_err;

  while (proposals_this_run < max_proposals &&
         (warmup_done < n_warmup || (n_prod_target > 0 && saved_confs < n_prod_target))) {
    proposals_this_run++;
    total_traj++;
    bool is_warmup = (warmup_done < n_warmup);
    if (is_warmup) warmup_done++;
    std::string phase_name = is_warmup ? "warmup" : "prod";

    unsigned long long seed_traj = (unsigned long long)hmc_momentum_seed + (unsigned long long)total_traj * seed_stride;
    hmc_param.momentum_seed = seed_traj;
    quda::setHMCBaseSeed(seed_traj);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Backup current host gauge buffer and pre-proposal plaquette/SHA256
    memcpy(gauge_bkup_buf.data(), gauge_buf.data(), gauge_bytes);
    std::string pre_sha256 = cur_sha256;
    double pre_plaq[3] = {cur_plaq[0], cur_plaq[1], cur_plaq[2]};

    double dH = 0.0;
    TrajectorySolveAudit aud;
    if (!run_monitored_hmc_trajectory(hmc_param, dH, aud)) {
      fprintf(stderr, "FATAL [Traj %d]: Solver/Hamiltonian health gate failed: %s\n",
              total_traj, aud.fail_reason.c_str());
      destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
      return 1;
    }

    double rand_val = quda::hmcMetropolisUniform();
    double prob = (dH < 0.0) ? 1.0 : std::exp(-dH);
    double raw_exp_minus_dH = std::exp(-dH);
    bool natural_accept = (rand_val < prob);
    bool forced_reject = (forced_reject_trajs.count(total_traj) > 0);
    bool accepted = forced_reject ? false : natural_accept;

    if (!accepted) {
      // Restore pre-proposal gauge field to device and refresh all derived resident fields
      loadGaugeQuda(hostGaugeBkup, &gauge_param);
      quda::hmcRefreshResidentGaugeState(inv_param);
      saveGaugeQuda(hostGauge, &gauge_param);
      std::string restored_sha = compute_sha256_buffer(gauge_buf.data(), gauge_bytes);
      double restored_plaq[3];
      plaqQuda(restored_plaq);
      double plaq_restore_diff = std::fabs(restored_plaq[0] - pre_plaq[0]);

      if (restored_sha != pre_sha256 || plaq_restore_diff >= 1e-14) {
        fprintf(stderr, "FATAL [Traj %d]: Rejection rollback verification failed! restored_sha=%s vs pre_sha=%s, |dPlaq|=%.3e\n",
                total_traj, restored_sha.c_str(), pre_sha256.c_str(), plaq_restore_diff);
        destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
        return 1;
      }
      multiplicity += 1;
      cur_sha256 = restored_sha;
      printf("Traj %d REJECTED (%s): restored state_id=%d (multiplicity=%d), |dPlaq_restore|=%.3e, sha256=%s\n",
             total_traj, forced_reject ? "TEST_FORCED_REJECT" : "METROPOLIS_REJECT",
             state_id, multiplicity, plaq_restore_diff, cur_sha256.substr(0, 12).c_str());
    } else {
      saveGaugeQuda(hostGauge, &gauge_param);
      cur_sha256 = compute_sha256_buffer(gauge_buf.data(), gauge_bytes);
      state_id += 1;
      multiplicity = 1;
    }

    // Read-only SU(3) unitarity and determinant check on active host gauge buffer
    check_su3_host_readonly(reinterpret_cast<const double *>(gauge_buf.data()), dim.data(), cur_u_err, cur_det_err, use_omp_su3);
    if (!std::isfinite(cur_u_err) || !std::isfinite(cur_det_err) || cur_u_err > 1e-12 || cur_det_err > 1e-12) {
      fprintf(stderr, "FATAL [Traj %d]: Read-only SU(3) check failed: max|U^dag U - I|=%.3e, max|det(U)-1|=%.3e (limit 1e-12)\n",
              total_traj, cur_u_err, cur_det_err);
      destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
      return 1;
    }

    // Compute plaquette and slow gauge observables (Polyakov loop, topological charge, energy) without SU(3) projection
    QudaGaugeObservableParam obs = newQudaGaugeObservableParam();
    obs.su_project = QUDA_BOOLEAN_FALSE;
    obs.compute_plaquette = QUDA_BOOLEAN_TRUE;
    obs.compute_polyakov_loop = QUDA_BOOLEAN_TRUE;
    obs.compute_qcharge = QUDA_BOOLEAN_TRUE;
    gaugeObservablesQuda(&obs);
    cur_plaq[0] = obs.plaquette[0];
    cur_plaq[1] = obs.plaquette[1];
    cur_plaq[2] = obs.plaquette[2];

    if (!std::isfinite(cur_plaq[0]) || !std::isfinite(cur_plaq[1]) || !std::isfinite(cur_plaq[2]) ||
        !std::isfinite(obs.ploop[0]) || !std::isfinite(obs.ploop[1]) ||
        !std::isfinite(obs.qcharge) || !std::isfinite(obs.energy[0])) {
      fprintf(stderr, "FATAL [Traj %d]: Non-finite gauge observable detected!\n", total_traj);
      destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
      return 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    long mem_avail = 0, vm_hwm = 0, swap_used = 0;
    get_memory_info(mem_avail, vm_hwm, swap_used);

    // Format ledger row
    std::ostringstream row_ss;
    row_ss << total_traj << "\t"
           << phase_name << "\t"
           << state_id << "\t"
           << multiplicity << "\t"
           << hmc_momentum_seed << "\t"
           << seed_traj << "\t"
           << std::scientific << std::setprecision(10)
           << aud.h_init << "\t"
           << aud.h_final << "\t"
           << dH << "\t"
           << raw_exp_minus_dH << "\t"
           << std::fixed << std::setprecision(10) << rand_val << "\t"
           << (accepted ? 1 : 0) << "\t"
           << (forced_reject ? 1 : 0) << "\t"
           << aud.parsed_solves << "\t"
           << aud.parsed_iter_sum << "\t"
           << aud.iter_max << "\t"
           << std::scientific << std::setprecision(6) << aud.max_true_res << "\t"
           << std::defaultfloat << std::setprecision(17)
           << cur_plaq[0] << "\t"
           << cur_plaq[1] << "\t"
           << cur_plaq[2] << "\t"
           << obs.ploop[0] << "\t"
           << obs.ploop[1] << "\t"
           << obs.qcharge << "\t"
           << obs.energy[0] << "\t"
           << std::scientific << std::setprecision(6)
           << cur_u_err << "\t"
           << cur_det_err << "\t"
           << std::fixed << std::setprecision(3) << elapsed << "\t"
           << std::fixed << std::setprecision(1)
           << (vm_hwm / 1024.0) << "\t"
           << (mem_avail / 1024.0) << "\t"
           << (swap_used / 1024.0) << "\t"
           << cur_sha256;
    ledger_lines.push_back(row_ss.str());

    // Handle production configuration saving
    if (!is_warmup && accepted) {
      accepted_prod_traj++;
      if (accepted_prod_traj % save_stride == 0) {
        saved_confs++;
        char conf_bin[512], conf_json[512];
        snprintf(conf_bin, sizeof(conf_bin), "%s/conf.%05d.bin", out_dir_opt.c_str(), saved_confs);
        snprintf(conf_json, sizeof(conf_json), "%s/conf.%05d.json", out_dir_opt.c_str(), saved_confs);

        if (!write_file_atomic_with_fsync(conf_bin, gauge_buf.data(), gauge_bytes)) {
          fprintf(stderr, "FATAL: Failed to write %s\n", conf_bin);
          destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
          return 1;
        }

        std::ostringstream cj;
        cj << std::setprecision(17);
        cj << "{\n"
           << "  \"configuration\": " << saved_confs << ",\n"
           << "  \"trajectory\": " << total_traj << ",\n"
           << "  \"phase\": \"prod\",\n"
           << "  \"state_id\": " << state_id << ",\n"
           << "  \"multiplicity\": " << multiplicity << ",\n"
           << "  \"geometry\": [" << dim[0] << ", " << dim[1] << ", " << dim[2] << ", " << dim[3] << "],\n"
           << "  \"gauge_order\": \"QUDA_QDP_GAUGE_ORDER\",\n"
           << "  \"fermion_t_boundary\": \"QUDA_ANTI_PERIODIC_T\",\n"
           << "  \"temporal_boundary_phase_slice\": \"dir=3,t=Lt-1,phase=-1\",\n"
           << "  \"ploop_convention\": \"QUDA_RAW_STORED_TEMPORAL_LINKS_INCLUDES_APBC_SIGN\",\n"
           << "  \"action_gauge\": \"Wilson plaquette\",\n"
           << "  \"action_fermion\": \"Two-degenerate-flavor (Nf=2) EO-preconditioned Wilson\",\n"
           << "  \"nf\": 2,\n"
           << "  \"beta\": " << hmc_beta << ",\n"
           << "  \"kappa\": " << kappa << ",\n"
           << "  \"csw\": " << csw_val << ",\n"
           << "  \"bare_mass\": " << inv_param.mass << ",\n"
           << "  \"integrator\": \"QUDA_OMELYAN_INTEGRATOR\",\n"
           << "  \"omelyan_lambda\": " << hmc_param.omelyan_lambda << ",\n"
           << "  \"tau\": " << hmc_tau << ",\n"
           << "  \"n_steps\": " << hmc_n_steps << ",\n"
           << "  \"dt\": " << (hmc_n_steps > 0 ? hmc_tau / hmc_n_steps : 0.0) << ",\n"
           << "  \"precision\": \"QUDA_DOUBLE_PRECISION\",\n"
           << "  \"reconstruct\": \"QUDA_RECONSTRUCT_NO\",\n"
           << "  \"solver_type\": \"QUDA_CG_INVERTER\",\n"
           << "  \"solver_tol\": " << inv_param.tol << ",\n"
           << "  \"solver_maxiter\": " << inv_param.maxiter << ",\n"
           << "  \"seed_base\": " << hmc_momentum_seed << ",\n"
           << "  \"seed_stride\": " << seed_stride << ",\n"
           << "  \"seed_trajectory\": " << seed_traj << ",\n"
           << "  \"H_init\": " << aud.h_init << ",\n"
           << "  \"H_final\": " << aud.h_final << ",\n"
           << "  \"dH\": " << dH << ",\n"
           << "  \"exp_minus_dH\": " << raw_exp_minus_dH << ",\n"
           << "  \"rand_val\": " << rand_val << ",\n"
           << "  \"accepted\": 1,\n"
           << "  \"solve_count\": " << aud.parsed_solves << ",\n"
           << "  \"iter_total\": " << aud.parsed_iter_sum << ",\n"
           << "  \"iter_max\": " << aud.iter_max << ",\n"
           << "  \"max_true_res\": " << aud.max_true_res << ",\n"
           << "  \"plaquette_total\": " << cur_plaq[0] << ",\n"
           << "  \"plaquette_spatial\": " << cur_plaq[1] << ",\n"
           << "  \"plaquette_temporal\": " << cur_plaq[2] << ",\n"
           << "  \"ploop_re\": " << obs.ploop[0] << ",\n"
           << "  \"ploop_im\": " << obs.ploop[1] << ",\n"
           << "  \"qcharge\": " << obs.qcharge << ",\n"
           << "  \"energy_total\": " << obs.energy[0] << ",\n"
           << "  \"su3_u_err\": " << cur_u_err << ",\n"
           << "  \"su3_det_err\": " << cur_det_err << ",\n"
           << "  \"sha256\": \"" << cur_sha256 << "\",\n"
           << "  \"bytes\": " << gauge_bytes << "\n"
           << "}\n";
        std::string cj_str = cj.str();
        if (!write_file_atomic_with_fsync(conf_json, cj_str.data(), cj_str.size())) {
          fprintf(stderr, "FATAL: Failed to write %s\n", conf_json);
          destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
          return 1;
        }
        printf("Saved Production Conf %05d (traj %d, state_id %d, sha256 %s)\n",
               saved_confs, total_traj, state_id, cur_sha256.substr(0, 12).c_str());
      }
    }

    // Publish crash-consistent checkpoint generation after EVERY trajectory (accepted or rejected)
    generation += 1;
    CheckpointManifest next_cm;
    next_cm.generation = generation;
    next_cm.total_traj = total_traj;
    next_cm.warmup_done = warmup_done;
    next_cm.accepted_prod_traj = accepted_prod_traj;
    next_cm.saved_confs = saved_confs;
    next_cm.state_id = state_id;
    next_cm.multiplicity = multiplicity;
    next_cm.phase = phase_name;
    for (int d = 0; d < 4; d++) next_cm.geometry[d] = dim[d];
    next_cm.beta = hmc_beta;
    next_cm.kappa = kappa;
    next_cm.csw = csw_val;
    next_cm.integrator = "QUDA_OMELYAN_INTEGRATOR";
    next_cm.tau = hmc_tau;
    next_cm.n_steps = hmc_n_steps;
    next_cm.solver_tol = inv_param.tol;
    next_cm.solver_maxiter = inv_param.maxiter;
    next_cm.seed_base = (unsigned long long)hmc_momentum_seed;
    next_cm.seed_stride = seed_stride;
    next_cm.last_seed_traj = seed_traj;
    next_cm.plaquette_total = cur_plaq[0];
    next_cm.plaquette_spatial = cur_plaq[1];
    next_cm.plaquette_temporal = cur_plaq[2];
    next_cm.ploop_re = obs.ploop[0];
    next_cm.ploop_im = obs.ploop[1];
    next_cm.qcharge = obs.qcharge;
    next_cm.energy_total = obs.energy[0];
    next_cm.su3_u_err = cur_u_err;
    next_cm.su3_det_err = cur_det_err;
    next_cm.gauge_sha256 = cur_sha256;

    if (!publish_checkpoint_generation(out_dir_opt, next_cm, gauge_buf.data(), gauge_bytes, ledger_lines)) {
      fprintf(stderr, "FATAL [Traj %d]: Failed to publish checkpoint generation %d\n", total_traj, generation);
      destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
      return 1;
    }

    auto t_complete_end = std::chrono::high_resolution_clock::now();
    double traj_complete_sec = std::chrono::duration<double>(t_complete_end - t_start).count();

    sum_traj_pre_io_sec += elapsed;
    sum_traj_complete_sec += traj_complete_sec;
    if (accepted) accepted_this_run++;
    sum_cg_iters += aud.parsed_iter_sum;
    if (aud.max_true_res > run_max_true_res) run_max_true_res = aud.max_true_res;
    if (cur_u_err > run_max_u_err) run_max_u_err = cur_u_err;
    if (cur_det_err > run_max_det_err) run_max_det_err = cur_det_err;

    printf("Traj %d (%s, state=%d, mult=%d, seed=%llu): dH=%+.6e, rand=%.6f, acc=%d (forced_rej=%d), solves=%d (iters=%d, max_res=%.3e), Plaq=%.17g, Q=%+.6e, u_err=%.3e, det_err=%.3e, Time=%.2fs, TimePreIO=%.3fs, TimeComplete=%.3fs, VmHWM=%.1fMB\n",
           total_traj, phase_name.c_str(), state_id, multiplicity, seed_traj,
           dH, rand_val, accepted ? 1 : 0, forced_reject ? 1 : 0,
           aud.parsed_solves, aud.parsed_iter_sum, aud.max_true_res,
           cur_plaq[0], obs.qcharge, cur_u_err, cur_det_err, elapsed, elapsed, traj_complete_sec, vm_hwm / 1024.0);
    fflush(stdout);
  }

  auto t_loop_end = std::chrono::high_resolution_clock::now();
  double end_to_end_wall_sec = std::chrono::duration<double>(t_loop_end - t_main_start).count();
  double steady_pre_io_mean = (proposals_this_run > 0) ? (sum_traj_pre_io_sec / proposals_this_run) : 0.0;
  double steady_complete_mean = (proposals_this_run > 0) ? (sum_traj_complete_sec / proposals_this_run) : 0.0;
  double steady_complete_traj_per_sec = (sum_traj_complete_sec > 0.0) ? (proposals_this_run / sum_traj_complete_sec) : 0.0;
  double end_to_end_complete_traj_per_sec = (end_to_end_wall_sec > 0.0) ? (proposals_this_run / end_to_end_wall_sec) : 0.0;
  double acceptance_rate = (proposals_this_run > 0) ? ((double)accepted_this_run / proposals_this_run) : 0.0;
  double accepted_complete_traj_per_sec = (sum_traj_complete_sec > 0.0) ? ((double)accepted_this_run / sum_traj_complete_sec) : 0.0;
  double end_to_end_accepted_traj_per_sec = (end_to_end_wall_sec > 0.0) ? ((double)accepted_this_run / end_to_end_wall_sec) : 0.0;

  printf("=== THROUGHPUT_SUMMARY_BEGIN ===\n");
  printf("PROPOSALS_THIS_RUN=%d\n", proposals_this_run);
  printf("ACCEPTED_THIS_RUN=%d\n", accepted_this_run);
  printf("ACCEPTANCE_RATE=%.6f\n", acceptance_rate);
  printf("SETUP_INIT_SEC=%.6f\n", setup_init_sec);
  printf("STEADY_STATE_PRE_IO_TRAJ_SEC_MEAN=%.6f\n", steady_pre_io_mean);
  printf("STEADY_STATE_COMPLETE_TRAJ_SEC_MEAN=%.6f\n", steady_complete_mean);
  printf("TOTAL_COMPLETE_TRAJ_SEC=%.6f\n", sum_traj_complete_sec);
  printf("END_TO_END_WALL_SEC=%.6f\n", end_to_end_wall_sec);
  printf("STEADY_STATE_COMPLETE_TRAJ_PER_SEC=%.6f\n", steady_complete_traj_per_sec);
  printf("END_TO_END_COMPLETE_TRAJ_PER_SEC=%.6f\n", end_to_end_complete_traj_per_sec);
  printf("ACCEPTED_COMPLETE_TRAJ_PER_SEC=%.6f\n", accepted_complete_traj_per_sec);
  printf("END_TO_END_ACCEPTED_COMPLETE_TRAJ_PER_SEC=%.6f\n", end_to_end_accepted_traj_per_sec);
  printf("MEAN_CG_ITERS_PER_TRAJ=%.2f\n", proposals_this_run > 0 ? (double)sum_cg_iters / proposals_this_run : 0.0);
  printf("MAX_TRUE_RES=%.6e\n", run_max_true_res);
  printf("MAX_SU3_U_ERR=%.6e\n", run_max_u_err);
  printf("MAX_SU3_DET_ERR=%.6e\n", run_max_det_err);
  printf("=== THROUGHPUT_SUMMARY_END ===\n");

  quda::logResidentMemoryBreakdown(V_sites, accum_pipeline);

  printf("=== Run Complete: total_traj=%d, warmup_done=%d, accepted_prod_traj=%d, saved_confs=%d, state_id=%d, mult=%d ===\n",
         total_traj, warmup_done, accepted_prod_traj, saved_confs, state_id, multiplicity);

  destroyHMCQuda();
  freeGaugeQuda();
  freeCloverQuda();
  endQuda();
  finalizeComms();
  return 0;
}
