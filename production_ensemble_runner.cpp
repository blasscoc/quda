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
#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <openssl/evp.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <quda.h>
#include <malloc_quda.h>
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

static void fadvise_dontneed_fd(int fd, off_t len = 0) {
  if (fd >= 0) {
#if defined(POSIX_FADV_DONTNEED)
    (void)posix_fadvise(fd, 0, len, POSIX_FADV_DONTNEED);
#endif
  }
}

static unsigned long long get_fs_free_bytes(const std::string &path) {
  struct statvfs st;
  if (statvfs(path.c_str(), &st) != 0) {
    return 0ULL;
  }
  return (unsigned long long)st.f_bavail * (unsigned long long)st.f_frsize;
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
  unsigned char buffer[65536];
  size_t bytes = 0;
  size_t total_read = 0;
  while ((bytes = fread(buffer, 1, sizeof(buffer), f)) != 0) {
    EVP_DigestUpdate(ctx, buffer, bytes);
    total_read += bytes;
  }
  if (total_read >= 1048576) {
    fadvise_dontneed_fd(fileno(f), 0);
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

[[maybe_unused]] static bool dir_exists(const std::string &path) {
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

static bool extract_json_object(const std::string &text, const std::string &key, std::string &out_obj_json) {
  std::string pat = "\"" + key + "\"";
  size_t pos = text.find(pat);
  if (pos == std::string::npos) return false;
  pos = text.find(':', pos + pat.size());
  if (pos == std::string::npos) return false;
  size_t lb = text.find('{', pos + 1);
  if (lb == std::string::npos) return false;
  int depth = 0;
  bool in_str = false;
  bool esc = false;
  for (size_t i = lb; i < text.size(); i++) {
    char c = text[i];
    if (in_str) {
      if (esc) {
        esc = false;
      } else if (c == '\\') {
        esc = true;
      } else if (c == '"') {
        in_str = false;
      }
    } else {
      if (c == '"') {
        in_str = true;
      } else if (c == '{') {
        depth++;
      } else if (c == '}') {
        depth--;
        if (depth == 0) {
          out_obj_json = text.substr(lb, i - lb + 1);
          return true;
        }
      }
    }
  }
  return false;
}

static volatile sig_atomic_t g_stop_requested = 0;
static volatile sig_atomic_t g_stop_signal = 0;
static void ensemble_stop_signal_handler(int sig) {
  g_stop_requested = 1;
  g_stop_signal = sig;
}

struct CheckpointManifest {
  int generation = 0;
  int total_traj = 0;
  int warmup_done = 0;
  int prod_traj = 0;
  int accepted_prod_traj = 0;
  int saved_confs = 0;
  int state_id = 0;
  int multiplicity = 1;
  std::string phase = "init";
  std::string init_provenance_json = "";
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
  if (extract_json_number(content, "prod_traj", v)) {
    cm.prod_traj = (int)std::llround(v);
  } else {
    cm.prod_traj = (cm.total_traj > cm.warmup_done) ? (cm.total_traj - cm.warmup_done) : 0;
  }
  if (!extract_json_number(content, "accepted_prod_traj", v)) { err_msg = "Missing 'accepted_prod_traj'"; return false; }
  cm.accepted_prod_traj = (int)std::llround(v);
  if (!extract_json_number(content, "saved_confs", v)) { err_msg = "Missing 'saved_confs'"; return false; }
  cm.saved_confs = (int)std::llround(v);
  if (!extract_json_number(content, "state_id", v)) { err_msg = "Missing 'state_id'"; return false; }
  cm.state_id = (int)std::llround(v);
  if (!extract_json_number(content, "multiplicity", v)) { err_msg = "Missing 'multiplicity'"; return false; }
  cm.multiplicity = (int)std::llround(v);
  extract_json_string(content, "phase", cm.phase);
  extract_json_object(content, "initialization_provenance", cm.init_provenance_json);

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
  fadvise_dontneed_fd(fileno(fg), 0);
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
  int fd = fileno(f);
  if (len >= (1ULL << 20)) {
    setvbuf(f, nullptr, _IONBF, 0);
  }
  const char *p = static_cast<const char *>(data);
  const size_t chunk_sz = 64ULL << 20; // 64 MiB chunks
  const size_t sync_evict_stride = 256ULL << 20; // sync + evict page cache every 256 MiB
  size_t written = 0;
  size_t since_evict = 0;
  while (written < len) {
    size_t to_write = (len - written > chunk_sz) ? chunk_sz : (len - written);
    if (fwrite(p + written, 1, to_write, f) != to_write) {
      fclose(f);
      unlink(tmp.c_str());
      return false;
    }
    written += to_write;
    since_evict += to_write;
    if (fd >= 0 && since_evict >= sync_evict_stride) {
      fflush(f);
      fdatasync(fd);
      fadvise_dontneed_fd(fd, written);
      since_evict = 0;
    }
  }
  if (fflush(f) != 0) {
    fclose(f);
    unlink(tmp.c_str());
    return false;
  }
  if (fd < 0 || fsync(fd) != 0) {
    fclose(f);
    unlink(tmp.c_str());
    return false;
  }
  if (len >= (1ULL << 20)) {
    fadvise_dontneed_fd(fd, 0);
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

static bool hardlink_file_atomic_with_fsync(const std::string &src_path, const std::string &dst_path) {
  std::string tmp = dst_path + ".tmp." + std::to_string(getpid());
  unlink(tmp.c_str());
  if (link(src_path.c_str(), tmp.c_str()) != 0) {
    return false;
  }
  if (rename(tmp.c_str(), dst_path.c_str()) != 0) {
    unlink(tmp.c_str());
    return false;
  }
  if (!fsync_parent_dir(dst_path)) {
    return false;
  }
  return true;
}

static unsigned long long sum_out_dir_regular_bytes(const std::string &out_dir) {
  DIR *dir = opendir(out_dir.c_str());
  if (!dir) return 0ULL;
  unsigned long long total = 0ULL;
  std::set<std::pair<dev_t, ino_t>> seen_inodes;
  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    if (name == "." || name == "..") continue;
    std::string full_path = out_dir + "/" + name;
    struct stat st;
    if (lstat(full_path.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
      std::pair<dev_t, ino_t> key(st.st_dev, st.st_ino);
      if (seen_inodes.insert(key).second) {
        total += (unsigned long long)st.st_size;
      }
    }
  }
  closedir(dir);
  return total;
}

static long get_host_mem_available_kb() {
  FILE *f = fopen("/proc/meminfo", "r");
  if (!f) return -1L;
  char line[256];
  long val = -1L;
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "MemAvailable:", 13) == 0) {
      sscanf(line + 13, "%ld", &val);
      break;
    }
  }
  fclose(f);
  return val;
}

static bool parse_ckpt_gen_filename(const std::string &name, int &gen_out) {
  if (name.compare(0, 9, "ckpt_gen_") != 0) return false;
  if (name.find(".tmp.") != std::string::npos) return false;
  if (name.find('/') != std::string::npos) return false;
  size_t dot_pos = name.find('.', 9);
  if (dot_pos == std::string::npos || dot_pos == 9) return false;
  for (size_t i = 9; i < dot_pos; i++) {
    if (name[i] < '0' || name[i] > '9') return false;
  }
  std::string suffix = name.substr(dot_pos);
  if (suffix != ".bin" && suffix != ".meta.json" && suffix != ".ledger.tsv") {
    return false;
  }
  int g = std::atoi(name.substr(9, dot_pos - 9).c_str());
  if (g < 1) return false;
  gen_out = g;
  return true;
}

static void prune_superseded_checkpoints(const std::string &out_dir, int current_gen, int keep_checkpoints) {
  if (keep_checkpoints < 1) return;
  int max_delete_gen = current_gen - keep_checkpoints;
  if (max_delete_gen < 1) return;

  DIR *dir = opendir(out_dir.c_str());
  if (!dir) return;
  std::vector<std::string> to_unlink;
  struct dirent *entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    int g = 0;
    if (parse_ckpt_gen_filename(name, g) && g <= max_delete_gen) {
      to_unlink.push_back(out_dir + "/" + name);
    }
  }
  closedir(dir);

  bool unlinked_any = false;
  for (const auto &fpath : to_unlink) {
    if (unlink(fpath.c_str()) == 0) {
      unlinked_any = true;
    }
  }
  if (unlinked_any) {
    fsync_parent_dir(out_dir + "/checkpoint_manifest.json");
  }
}

static bool publish_checkpoint_generation(const std::string &out_dir,
                                          CheckpointManifest &cm,
                                          const void *gauge_data,
                                          size_t gauge_bytes,
                                          const std::vector<std::string> &ledger_lines,
                                          int keep_checkpoints = 2,
                                          unsigned long long max_out_dir_bytes = 0ULL,
                                          unsigned long long min_free_disk_bytes = 107374182400ULL) {
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

  std::ostringstream l_ss;
  for (const auto &ln : ledger_lines) l_ss << ln << "\n";
  std::string l_str = l_ss.str();

  unsigned long long cur_dir_bytes = sum_out_dir_regular_bytes(out_dir);
  unsigned long long ckpt_peak_bytes = (unsigned long long)gauge_bytes + 2ULL * (unsigned long long)l_str.size() + 16384ULL;
  if (max_out_dir_bytes > 0ULL && cur_dir_bytes + ckpt_peak_bytes > max_out_dir_bytes) {
    fprintf(stderr,
            "FATAL [Gen %d]: Pre-write storage guard in publish_checkpoint_generation triggered for '%s': "
            "current_regular_bytes=%llu + ckpt_transient_bytes=%llu = projected_peak_bytes=%llu > max_out_dir_bytes=%llu. "
            "Refusing to write; previous checkpoint generation remains intact.\n",
            cm.generation, out_dir.c_str(), cur_dir_bytes, ckpt_peak_bytes,
            cur_dir_bytes + ckpt_peak_bytes, max_out_dir_bytes);
    return false;
  }
  if (min_free_disk_bytes > 0ULL) {
    unsigned long long avail_disk_bytes = get_fs_free_bytes(out_dir);
    if (avail_disk_bytes > 0ULL && avail_disk_bytes < ckpt_peak_bytes + min_free_disk_bytes) {
      fprintf(stderr,
              "FATAL [Gen %d]: Filesystem free-disk guard in publish_checkpoint_generation triggered for '%s': "
              "avail_disk_bytes=%llu < ckpt_transient_bytes=%llu + min_free_disk_bytes=%llu. "
              "Refusing to write; previous checkpoint generation remains intact.\n",
              cm.generation, out_dir.c_str(), avail_disk_bytes, ckpt_peak_bytes, min_free_disk_bytes);
      return false;
    }
  }

  std::string g_path = out_dir + "/" + g_name;
  if (!write_file_atomic_with_fsync(g_path, gauge_data, gauge_bytes)) return false;

  std::string l_path = out_dir + "/" + l_name;
  if (!write_file_atomic_with_fsync(l_path, l_str.data(), l_str.size())) return false;
  cm.ledger_sha256 = compute_sha256_buffer(l_str.data(), l_str.size());

  if (cm.init_provenance_json.empty()) {
    cm.init_provenance_json = "{\"kind\": \"cold_unit_gauge\", \"thermalized_at_target_volume\": false}";
  }
  std::ostringstream m_ss;
  m_ss << std::setprecision(17);
  m_ss << "{\n"
       << "  \"generation\": " << cm.generation << ",\n"
       << "  \"total_traj\": " << cm.total_traj << ",\n"
       << "  \"warmup_done\": " << cm.warmup_done << ",\n"
       << "  \"prod_traj\": " << cm.prod_traj << ",\n"
       << "  \"accepted_prod_traj\": " << cm.accepted_prod_traj << ",\n"
       << "  \"saved_confs\": " << cm.saved_confs << ",\n"
       << "  \"phase\": \"" << cm.phase << "\",\n"
       << "  \"state_id\": " << cm.state_id << ",\n"
       << "  \"multiplicity\": " << cm.multiplicity << ",\n"
       << "  \"initialization_provenance\": " << cm.init_provenance_json << ",\n"
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
         << "  \"prod_traj\": " << cm.prod_traj << ",\n"
         << "  \"accepted_prod_traj\": " << cm.accepted_prod_traj << ",\n"
         << "  \"saved_confs\": " << cm.saved_confs << ",\n"
         << "  \"state_id\": " << cm.state_id << ",\n"
         << "  \"multiplicity\": " << cm.multiplicity << ",\n"
         << "  \"phase\": \"" << cm.phase << "\",\n"
         << "  \"initialization_provenance\": " << cm.init_provenance_json << ",\n"
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
  if (!write_file_atomic_with_fsync(main_ledger_path, l_str.data(), l_str.size())) return false;
  prune_superseded_checkpoints(out_dir, cm.generation, keep_checkpoints);
  return true;
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
  int keep_checkpoints = 2;
  unsigned long long max_out_dir_bytes = 0ULL;
  unsigned long long min_free_disk_bytes = 107374182400ULL;

  app->add_option("--mode", mode, "Execution mode: run | reversibility | validate-checkpoint | multi-rhs-probe | bench_fp4_iter");
  app->add_option("--warmup", n_warmup, "Number of warmup proposals");
  app->add_option("--prod", n_prod_target, "Target accepted production configurations");
  app->add_option("--max-proposals", max_proposals, "Maximum proposals in this invocation");
  app->add_option("--stride", save_stride, "Save stride in accepted production trajectories");
  app->add_option("--out-dir", out_dir_opt, "Output directory");
  app->add_option("--keep-checkpoints", keep_checkpoints, "Number of newest completed ckpt_gen_* generations to retain in out_dir (default 2, must be >= 1)");
  app->add_option("--max-out-dir-bytes", max_out_dir_bytes, "Optional pre-write storage budget in bytes for out_dir including transient .tmp peak (default 0 = unlimited)");
  app->add_option("--min-free-disk-bytes", min_free_disk_bytes, "Minimum free filesystem bytes required after transient writes via statvfs (default 107374182400 = 100 GiB, 0 = disabled)");
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
  std::string init_provenance_json_opt = "";
  app->add_option("--init-provenance-json", init_provenance_json_opt,
                  "Explicit JSON object string recording initialization provenance (e.g. cold_unit_gauge or tiled_checkpoint)");
  std::string init_gauge_bin_opt = "";
  app->add_option("--init-gauge-bin", init_gauge_bin_opt,
                  "Optional raw QDP FP64 gauge binary to load at trajectory 0 (e.g. periodic-tiled checkpoint)");

  try {
    app->parse(argc, argv);
  } catch (const CLI::ParseError &e) {
    return app->exit(e);
  }

  if (keep_checkpoints < 1) {
    fprintf(stderr, "FATAL: --keep-checkpoints must be >= 1 (got %d)\n", keep_checkpoints);
    return 1;
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

  // Avoid allocating 2,504 B/site (24.76 GiB at 48^3x96, 60.45 GiB at 60^3x120) in host_rand
  // inside initComms() -> initRand() when hot host-random gauge initialization is not used.
  const int req_dim[4] = {dim[0], dim[1], dim[2], dim[3]};
  bool need_host_rand = (mode == "bench_fp4_iter") ||
                        (!has_existing_checkpoint && init_gauge_bin_opt.empty() && !unit_gauge);
  const auto saved_dim = dim;
  if (!need_host_rand) {
    dim[0] = 2;
    dim[1] = 2;
    dim[2] = 2;
    dim[3] = 2;
  }
  int dummy_argc = 1;
  char prog_name[] = "production_ensemble_runner";
  char *dummy_argv[] = {prog_name, nullptr};
  initComms(dummy_argc, dummy_argv, gridsize_from_cmdline);
  dim = saved_dim;
  if (!need_host_rand) {
    host_rand.clear();
    host_rand.shrink_to_fit();
  }

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
  if (dim[0] != req_dim[0] || dim[1] != req_dim[1] || dim[2] != req_dim[2] || dim[3] != req_dim[3] ||
      gauge_param.X[0] != req_dim[0] || gauge_param.X[1] != req_dim[1] ||
      gauge_param.X[2] != req_dim[2] || gauge_param.X[3] != req_dim[3] ||
      V != (req_dim[0] * req_dim[1] * req_dim[2] * req_dim[3])) {
    fprintf(stderr,
            "FATAL: Runtime lattice dimensions (%dx%dx%dx%d, gauge_param.X=%dx%dx%dx%d, V=%d) do not match requested CLI dimensions (%dx%dx%dx%d)!\n",
            dim[0], dim[1], dim[2], dim[3],
            gauge_param.X[0], gauge_param.X[1], gauge_param.X[2], gauge_param.X[3], V,
            req_dim[0], req_dim[1], req_dim[2], req_dim[3]);
    return 1;
  }
  printf("RUNTIME_DIM_CHECK_PASS: requested=%dx%dx%dx%d runtime=%dx%dx%dx%d V=%d host_rand_size=%zu\n",
         req_dim[0], req_dim[1], req_dim[2], req_dim[3],
         gauge_param.X[0], gauge_param.X[1], gauge_param.X[2], gauge_param.X[3],
         V, host_rand.size());

  initQuda(device_ordinal);
  setVerbosityQuda(QUDA_SUMMARIZE, "", stdout);
  if (need_host_rand) {
    initRand();
  }

  size_t V_sites = (size_t)gauge_param.X[0] * gauge_param.X[1] * gauge_param.X[2] * gauge_param.X[3];
  size_t gauge_bytes = 4 * V_sites * 18 * sizeof(double);
  std::vector<char> gauge_buf;
  bool gauge_buf_preloaded_from_resume = false;
  if (!resumed_gauge_buf.empty() && resumed_gauge_buf.size() == gauge_bytes) {
    gauge_buf = std::move(resumed_gauge_buf);
    std::vector<char>().swap(resumed_gauge_buf);
    gauge_buf_preloaded_from_resume = true;
  } else {
    gauge_buf.resize(gauge_bytes);
  }
  bool need_host_bkup = (mode == "reversibility" || mode == "bench_fp4_iter");
  std::vector<char> gauge_bkup_buf(need_host_bkup ? gauge_bytes : 0);
  void *hostGauge[4], *hostGaugeBkup[4];
  for (int d = 0; d < 4; d++) {
    hostGauge[d] = gauge_buf.data() + d * V_sites * 18 * sizeof(double);
    hostGaugeBkup[d] = need_host_bkup ? (gauge_bkup_buf.data() + d * V_sites * 18 * sizeof(double)) : nullptr;
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
        size_t nread = fread(&cap_text[0], 1, fsz, cap_fp);
        cap_text.resize(nread);
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
      if (!gauge_buf_preloaded_from_resume) {
        memcpy(gauge_buf.data(), resumed_gauge_buf.data(), gauge_bytes);
        std::vector<char>().swap(resumed_gauge_buf);
      }
    } else {
      constructHostGaugeField(hostGauge, gauge_param, dummy_argc, dummy_argv);
      std::vector<std::mt19937_64>().swap(host_rand);
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
      if (!gauge_buf_preloaded_from_resume) {
        memcpy(gauge_buf.data(), resumed_gauge_buf.data(), gauge_bytes);
        std::vector<char>().swap(resumed_gauge_buf);
      }
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
      std::vector<std::mt19937_64>().swap(host_rand);
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
    double rel_dH_rev = abs_dH_rev / std::max(1.0, std::fabs(aud_fwd.h_init));
    double abs_dPlaq_rev = std::fabs(plaq_return[0] - plaq_start[0]);

    std::string rev_json_path = out_dir_opt + "/reversibility_result.json";
    FILE *fr = fopen(rev_json_path.c_str(), "w");
    bool pass = std::isfinite(abs_dH_rev) && std::isfinite(rel_dH_rev) && std::isfinite(abs_dPlaq_rev) &&
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
      fprintf(fr, "  \"rel_H_return_minus_H_start\": %.12e,\n", rel_dH_rev);
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
  int prod_traj = 0;
  int accepted_prod_traj = 0;
  int saved_confs = 0;
  int state_id = 0;
  int multiplicity = 1;
  std::string active_init_provenance_json = init_provenance_json_opt;

  struct sigaction sa_stop;
  memset(&sa_stop, 0, sizeof(sa_stop));
  sa_stop.sa_handler = ensemble_stop_signal_handler;
  sigemptyset(&sa_stop.sa_mask);
  sa_stop.sa_flags = 0;
  sigaction(SIGTERM, &sa_stop, nullptr);
  sigaction(SIGINT, &sa_stop, nullptr);
  sigaction(SIGUSR1, &sa_stop, nullptr);

  std::string pid_file_path = out_dir_opt + "/runner.pid";
  std::string stop_sentinel_path = out_dir_opt + "/STOP";
  std::string stop_req_sentinel_path = out_dir_opt + "/STOP_REQUESTED";
  std::string dot_stop_sentinel_path = out_dir_opt + "/.stop_requested";
  auto has_stop_sentinel = [&]() -> bool {
    return file_exists(stop_sentinel_path) || file_exists(stop_req_sentinel_path) || file_exists(dot_stop_sentinel_path);
  };
  auto clear_stop_sentinels = [&]() {
    unlink(stop_sentinel_path.c_str());
    unlink(stop_req_sentinel_path.c_str());
    unlink(dot_stop_sentinel_path.c_str());
  };
  {
    std::string pid_str = std::to_string(getpid()) + "\n";
    write_file_atomic_with_fsync(pid_file_path, pid_str.data(), pid_str.size());
  }

  if (has_existing_checkpoint) {
    generation = cm.generation;
    total_traj = cm.total_traj;
    warmup_done = cm.warmup_done;
    prod_traj = cm.prod_traj;
    accepted_prod_traj = cm.accepted_prod_traj;
    saved_confs = cm.saved_confs;
    state_id = cm.state_id;
    multiplicity = cm.multiplicity;
    if (active_init_provenance_json.empty()) {
      active_init_provenance_json = cm.init_provenance_json;
    }
    if (!gauge_buf_preloaded_from_resume) {
      memcpy(gauge_buf.data(), resumed_gauge_buf.data(), gauge_bytes);
      std::vector<char>().swap(resumed_gauge_buf);
    }

    std::ifstream fl(out_dir_opt + "/" + cm.ledger_file);
    std::string ln;
    while (std::getline(fl, ln)) {
      if (!ln.empty()) ledger_lines.push_back(ln);
    }
    printf("Resumed verified checkpoint generation %d (total_traj=%d, warmup_done=%d, prod_traj=%d, accepted_prod_traj=%d, saved_confs=%d, state_id=%d, mult=%d, sha256=%s)\n",
           generation, total_traj, warmup_done, prod_traj, accepted_prod_traj, saved_confs, state_id, multiplicity,
           cm.gauge_sha256.substr(0, 12).c_str());
  } else {
    if (!init_gauge_bin_opt.empty()) {
      FILE *fg = fopen(init_gauge_bin_opt.c_str(), "rb");
      if (!fg) {
        fprintf(stderr, "FATAL: Cannot open --init-gauge-bin '%s'\n", init_gauge_bin_opt.c_str());
        destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
        return 1;
      }
      if (fseeko(fg, 0, SEEK_END) != 0) {
        fclose(fg);
        fprintf(stderr, "FATAL: Cannot seek --init-gauge-bin '%s'\n", init_gauge_bin_opt.c_str());
        destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
        return 1;
      }
      off_t fsz = ftello(fg);
      if ((size_t)fsz != gauge_bytes) {
        fclose(fg);
        fprintf(stderr, "FATAL: --init-gauge-bin '%s' size mismatch: got %lld bytes, expected %zu bytes\n",
                init_gauge_bin_opt.c_str(), (long long)fsz, gauge_bytes);
        destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
        return 1;
      }
      fseeko(fg, 0, SEEK_SET);
      size_t rb = fread(gauge_buf.data(), 1, gauge_bytes, fg);
      fadvise_dontneed_fd(fileno(fg), 0);
      fclose(fg);
      if (rb != gauge_bytes) {
        fprintf(stderr, "FATAL: Short read on --init-gauge-bin '%s'\n", init_gauge_bin_opt.c_str());
        destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
        return 1;
      }
      printf("Loaded initial gauge field from '%s' (%zu bytes, page-cache evicted via POSIX_FADV_DONTNEED)\n",
             init_gauge_bin_opt.c_str(), gauge_bytes);
    } else {
      constructHostGaugeField(hostGauge, gauge_param, dummy_argc, dummy_argv);
      std::vector<std::mt19937_64>().swap(host_rand);
    }
    ledger_lines.push_back(get_ledger_header());
    if (active_init_provenance_json.empty()) {
      active_init_provenance_json = "{\"kind\": \"cold_unit_gauge\", \"thermalized_at_target_volume\": false}";
    }
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
  bool stopped_gracefully = false;

  // Recover last saved configuration state_id and sha256 if confs already exist on resume
  int last_saved_conf_idx = saved_confs;
  int last_saved_state_id = -1;
  std::string last_saved_sha256 = "";
  if (saved_confs > 0) {
    char prev_conf_json[512];
    snprintf(prev_conf_json, sizeof(prev_conf_json), "%s/conf.%05d.json", out_dir_opt.c_str(), saved_confs);
    std::ifstream fprev(prev_conf_json);
    if (fprev) {
      std::stringstream ssprev;
      ssprev << fprev.rdbuf();
      std::string prev_txt = ssprev.str();
      double v_sid = -1.0;
      if (extract_json_number(prev_txt, "state_id", v_sid)) {
        last_saved_state_id = (int)std::llround(v_sid);
      }
      extract_json_string(prev_txt, "sha256", last_saved_sha256);
    }
  }

  while (proposals_this_run < max_proposals &&
         (warmup_done < n_warmup || (n_prod_target > 0 && saved_confs < n_prod_target))) {
    if (g_stop_requested || has_stop_sentinel()) {
      printf("Graceful stop requested before starting next trajectory (signal=%d, sentinel=%d); stopping at generation=%d, total_traj=%d.\n",
             (int)g_stop_signal, has_stop_sentinel() ? 1 : 0, generation, total_traj);
      clear_stop_sentinels();
      stopped_gracefully = true;
      break;
    }
    proposals_this_run++;
    total_traj++;
    bool is_warmup = (warmup_done < n_warmup);
    if (is_warmup) {
      warmup_done++;
    } else {
      prod_traj++;
    }
    std::string phase_name = is_warmup ? "warmup" : "prod";

    unsigned long long seed_traj = (unsigned long long)hmc_momentum_seed + (unsigned long long)total_traj * seed_stride;
    hmc_param.momentum_seed = seed_traj;
    quda::setHMCBaseSeed(seed_traj);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Pre-proposal plaquette/SHA256: gauge_buf already holds U_old (not mutated during run_monitored_hmc_trajectory)
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
      // Restore pre-proposal gauge field directly from gauge_buf (which still holds U_old) to device
      loadGaugeQuda(hostGauge, &gauge_param);
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
    quda::pool::flush_device();
    gaugeObservablesQuda(&obs);
    quda::pool::flush_device();
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

    if (!is_warmup && accepted) {
      accepted_prod_traj++;
    }

    // Unbiased Markov-chain proposal cadence: save every save_stride production proposals (accepted OR rejected).
    // If a proposal is rejected and the active state_id matches the previously saved configuration, we hardlink
    // conf.%05d.bin to the previous configuration binary so zero duplicate gauge bytes are consumed on disk while
    // preserving exact 1-to-1 unweighted conf.%05d.{bin,json} Markov sample indices AND explicit residence weights.
    bool will_save_conf = (!is_warmup && (prod_traj % save_stride == 0));
    bool can_hardlink_prev_conf = false;
    int hardlink_source_conf_idx = 0;
    if (will_save_conf && !accepted && last_saved_conf_idx > 0 &&
        last_saved_state_id == state_id && last_saved_sha256 == cur_sha256) {
      char prev_bin_check[512];
      snprintf(prev_bin_check, sizeof(prev_bin_check), "%s/conf.%05d.bin", out_dir_opt.c_str(), last_saved_conf_idx);
      if (file_exists(prev_bin_check)) {
        can_hardlink_prev_conf = true;
        hardlink_source_conf_idx = last_saved_conf_idx;
      }
    }

    // Hard pre-write storage budget guard for out_dir before writing conf.*.bin or ckpt_gen_*.bin
    size_t est_ledger_bytes = 0;
    for (const auto &ln : ledger_lines) est_ledger_bytes += ln.size() + 1;
    unsigned long long cur_out_dir_bytes = sum_out_dir_regular_bytes(out_dir_opt);
    unsigned long long conf_transient_bytes = 0ULL;
    if (will_save_conf) {
      conf_transient_bytes = can_hardlink_prev_conf ? 4096ULL : ((unsigned long long)gauge_bytes + 4096ULL);
    }
    unsigned long long ckpt_transient_bytes = (unsigned long long)gauge_bytes + 2ULL * (unsigned long long)est_ledger_bytes + 16384ULL;
    unsigned long long projected_peak_bytes = cur_out_dir_bytes + conf_transient_bytes + ckpt_transient_bytes;
    if (max_out_dir_bytes > 0ULL && projected_peak_bytes > max_out_dir_bytes) {
      fprintf(stderr,
              "FATAL [Traj %d]: Pre-write storage guard triggered for '%s': "
              "current_regular_bytes=%llu + transient_peak_bytes=%llu (conf_write=%llu, ckpt_write=%llu) "
              "= projected_peak_bytes=%llu > max_out_dir_bytes=%llu. "
              "Refusing to write; previous checkpoint generation %d remains intact and valid.\n",
              total_traj, out_dir_opt.c_str(), cur_out_dir_bytes,
              conf_transient_bytes + ckpt_transient_bytes, conf_transient_bytes, ckpt_transient_bytes,
              projected_peak_bytes, max_out_dir_bytes, generation);
      unlink(pid_file_path.c_str());
      destroyHMCQuda();
      freeGaugeQuda();
      endQuda();
      finalizeComms();
      return 2;
    }
    if (min_free_disk_bytes > 0ULL) {
      unsigned long long avail_disk_bytes = get_fs_free_bytes(out_dir_opt);
      unsigned long long transient_need = conf_transient_bytes + ckpt_transient_bytes;
      if (avail_disk_bytes > 0ULL && avail_disk_bytes < transient_need + min_free_disk_bytes) {
        fprintf(stderr,
                "FATAL [Traj %d]: Filesystem free-disk guard triggered for '%s': "
                "avail_disk_bytes=%llu < transient_peak_bytes=%llu + min_free_disk_bytes=%llu. "
                "Refusing to write; previous checkpoint generation %d remains intact and valid.\n",
                total_traj, out_dir_opt.c_str(), avail_disk_bytes,
                transient_need, min_free_disk_bytes, generation);
        unlink(pid_file_path.c_str());
        destroyHMCQuda();
        freeGaugeQuda();
        endQuda();
        finalizeComms();
        return 2;
      }
    }

    // Handle production configuration saving on true Markov proposal cadence (prod_traj % save_stride == 0)
    if (will_save_conf) {
      saved_confs++;
      char conf_bin[512], conf_json[512];
      snprintf(conf_bin, sizeof(conf_bin), "%s/conf.%05d.bin", out_dir_opt.c_str(), saved_confs);
      snprintf(conf_json, sizeof(conf_json), "%s/conf.%05d.json", out_dir_opt.c_str(), saved_confs);

      std::string hardlinked_from_str = "";
      if (can_hardlink_prev_conf) {
        char prev_conf_bin[512];
        snprintf(prev_conf_bin, sizeof(prev_conf_bin), "%s/conf.%05d.bin", out_dir_opt.c_str(), hardlink_source_conf_idx);
        char rel_prev_name[64];
        snprintf(rel_prev_name, sizeof(rel_prev_name), "conf.%05d.bin", hardlink_source_conf_idx);
        if (hardlink_file_atomic_with_fsync(prev_conf_bin, conf_bin)) {
          hardlinked_from_str = rel_prev_name;
        } else {
          if (!write_file_atomic_with_fsync(conf_bin, gauge_buf.data(), gauge_bytes)) {
            fprintf(stderr, "FATAL: Failed to write %s\n", conf_bin);
            unlink(pid_file_path.c_str());
            destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
            return 1;
          }
        }
      } else {
        if (!write_file_atomic_with_fsync(conf_bin, gauge_buf.data(), gauge_bytes)) {
          fprintf(stderr, "FATAL: Failed to write %s\n", conf_bin);
          unlink(pid_file_path.c_str());
          destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
          return 1;
        }
      }

      std::string sample_kind = accepted ? "accepted_proposal" : "rejected_repeat";
      int origin_conf = can_hardlink_prev_conf ? hardlink_source_conf_idx : saved_confs;

      std::ostringstream cj;
      cj << std::setprecision(17);
      cj << "{\n"
         << "  \"configuration\": " << saved_confs << ",\n"
         << "  \"trajectory\": " << total_traj << ",\n"
         << "  \"prod_traj\": " << prod_traj << ",\n"
         << "  \"accepted_prod_traj\": " << accepted_prod_traj << ",\n"
         << "  \"phase\": \"prod\",\n"
         << "  \"sample_cadence\": \"markov_proposal_stride\",\n"
         << "  \"save_stride\": " << save_stride << ",\n"
         << "  \"sample_kind\": \"" << sample_kind << "\",\n"
         << "  \"state_id\": " << state_id << ",\n"
         << "  \"multiplicity\": " << multiplicity << ",\n"
         << "  \"origin_configuration\": " << origin_conf << ",\n"
         << "  \"hardlinked_from\": \"" << hardlinked_from_str << "\",\n"
         << "  \"initialization_provenance\": " << active_init_provenance_json << ",\n"
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
         << "  \"accepted\": " << (accepted ? 1 : 0) << ",\n"
         << "  \"forced_reject\": " << (forced_reject ? 1 : 0) << ",\n"
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
        unlink(pid_file_path.c_str());
        destroyHMCQuda(); freeGaugeQuda(); endQuda(); finalizeComms();
        return 1;
      }
      last_saved_conf_idx = saved_confs;
      last_saved_state_id = state_id;
      last_saved_sha256 = cur_sha256;
      printf("Saved Production Conf %05d (traj %d, prod_traj %d, acc=%d, kind=%s, state_id %d, mult %d, hardlink=%s, sha256 %s)\n",
             saved_confs, total_traj, prod_traj, accepted ? 1 : 0, sample_kind.c_str(),
             state_id, multiplicity, hardlinked_from_str.empty() ? "none" : hardlinked_from_str.c_str(),
             cur_sha256.substr(0, 12).c_str());
    }

    // Publish crash-consistent checkpoint generation after EVERY trajectory (accepted or rejected)
    generation += 1;
    CheckpointManifest next_cm;
    next_cm.generation = generation;
    next_cm.total_traj = total_traj;
    next_cm.warmup_done = warmup_done;
    next_cm.prod_traj = prod_traj;
    next_cm.accepted_prod_traj = accepted_prod_traj;
    next_cm.saved_confs = saved_confs;
    next_cm.state_id = state_id;
    next_cm.multiplicity = multiplicity;
    next_cm.phase = phase_name;
    next_cm.init_provenance_json = active_init_provenance_json;
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

    if (!publish_checkpoint_generation(out_dir_opt, next_cm, gauge_buf.data(), gauge_bytes, ledger_lines, keep_checkpoints, max_out_dir_bytes, min_free_disk_bytes)) {
      fprintf(stderr, "FATAL [Traj %d]: Failed to publish checkpoint generation %d\n", total_traj, generation);
      unlink(pid_file_path.c_str());
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

    long mem_avail_kb = get_host_mem_available_kb();
    printf("Traj %d (%s, state=%d, mult=%d, seed=%llu): dH=%+.6e, rand=%.6f, acc=%d (forced_rej=%d), solves=%d (iters=%d, max_res=%.3e), Plaq=%.17g, Q=%+.6e, u_err=%.3e, det_err=%.3e, Time=%.2fs, TimePreIO=%.3fs, TimeComplete=%.3fs, VmHWM=%.1fMB, MemAvail=%.1fMB\n",
           total_traj, phase_name.c_str(), state_id, multiplicity, seed_traj,
           dH, rand_val, accepted ? 1 : 0, forced_reject ? 1 : 0,
           aud.parsed_solves, aud.parsed_iter_sum, aud.max_true_res,
           cur_plaq[0], obs.qcharge, cur_u_err, cur_det_err, elapsed, elapsed, traj_complete_sec, vm_hwm / 1024.0, mem_avail_kb / 1024.0);
    fflush(stdout);

    if (g_stop_requested || has_stop_sentinel()) {
      printf("Graceful stop completed after publishing checkpoint generation %d (total_traj=%d, state_id=%d, mult=%d, signal=%d, sentinel=%d).\n",
             generation, total_traj, state_id, multiplicity, (int)g_stop_signal, has_stop_sentinel() ? 1 : 0);
      fflush(stdout);
      clear_stop_sentinels();
      stopped_gracefully = true;
      break;
    }
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

  printf("=== Run %s: total_traj=%d, warmup_done=%d, prod_traj=%d, accepted_prod_traj=%d, saved_confs=%d, state_id=%d, mult=%d ===\n",
         stopped_gracefully ? "Stopped Gracefully" : "Complete",
         total_traj, warmup_done, prod_traj, accepted_prod_traj, saved_confs, state_id, multiplicity);

  unlink(pid_file_path.c_str());
  destroyHMCQuda();
  freeGaugeQuda();
  freeCloverQuda();
  endQuda();
  finalizeComms();
  return 0;
}
