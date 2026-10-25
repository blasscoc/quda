#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <complex>
#include <limits>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <openssl/evp.h>

#include <quda.h>
#include <gauge_field.h>
#include <gauge_tools.h>
#include "command_line_params.h"
#include "gauge_utils.h"
#include "host_utils.h"

static bool dir_exists(const std::string &path) {
  struct stat st;
  return (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

static bool file_exists(const std::string &path) {
  struct stat st;
  return (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode));
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

static bool is_valid_sha256_hex(const std::string &s) {
  if (s.size() != 64) return false;
  for (char c : s) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
      return false;
    }
  }
  return true;
}

static void check_su3_host_readonly(const double *buf, const int X[4],
                                    double &max_unitarity_err, double &max_det_err) {
  max_unitarity_err = 0.0;
  max_det_err = 0.0;
  size_t V_sites = (size_t)X[0] * (size_t)X[1] * (size_t)X[2] * (size_t)X[3];
  size_t Vh_sites = V_sites / 2;
  size_t lower_bound = ((size_t)(X[0] / 2)) * (size_t)X[1] * (size_t)X[2] * (size_t)(X[3] - 1);

  using cd = std::complex<double>;
  for (int dir = 0; dir < 4; dir++) {
    for (int parity = 0; parity < 2; parity++) {
      for (size_t j = 0; j < Vh_sites; j++) {
        size_t m = (size_t)dir * V_sites + (size_t)parity * Vh_sites + j;
        const double *p = buf + m * 18;
        for (int k = 0; k < 18; k++) {
          if (!std::isfinite(p[k])) {
            max_unitarity_err = std::numeric_limits<double>::quiet_NaN();
            max_det_err = std::numeric_limits<double>::quiet_NaN();
            return;
          }
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
              max_unitarity_err = std::numeric_limits<double>::quiet_NaN();
              max_det_err = std::numeric_limits<double>::quiet_NaN();
              return;
            }
            if (err > max_unitarity_err) max_unitarity_err = err;
          }
        }
        cd det = U[0][0] * (U[1][1] * U[2][2] - U[1][2] * U[2][1])
               - U[0][1] * (U[1][0] * U[2][2] - U[1][2] * U[2][0])
               + U[0][2] * (U[1][0] * U[2][1] - U[1][1] * U[2][0]);
        double derr = std::abs(det - cd(1.0, 0.0));
        if (!std::isfinite(derr)) {
          max_unitarity_err = std::numeric_limits<double>::quiet_NaN();
          max_det_err = std::numeric_limits<double>::quiet_NaN();
          return;
        }
        if (derr > max_det_err) max_det_err = derr;
      }
    }
  }
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
  if (!(iss >> X[0] >> X[1] >> X[2] >> X[3])) return false;
  return (X[0] > 0 && X[1] > 0 && X[2] > 0 && X[3] > 0);
}

struct FieldTarget {
  std::string bin_path;
  std::string json_path;
  int X[4] = {0, 0, 0, 0};
  size_t expected_bytes = 0;
  std::string expected_sha256;
  double expected_plaq_total = 0.0;
  double expected_plaq_spatial = 0.0;
  double expected_plaq_temporal = 0.0;
  double expected_ploop_re = 0.0;
  double expected_ploop_im = 0.0;
  double expected_qcharge = 0.0;
  double expected_energy_total = 0.0;
  double expected_su3_u_err = 0.0;
  double expected_su3_det_err = 0.0;
};

static bool load_field_target_from_json(const std::string &json_path, FieldTarget &ft, std::string &err_msg) {
  std::ifstream fin(json_path);
  if (!fin) {
    err_msg = "cannot open JSON file: " + json_path;
    return false;
  }
  std::stringstream ss;
  ss << fin.rdbuf();
  std::string content = ss.str();

  ft.json_path = json_path;
  if (!extract_json_geometry(content, ft.X)) {
    err_msg = "missing or invalid 'geometry' in " + json_path;
    return false;
  }

  double b_d = 0.0;
  if (!extract_json_number(content, "bytes", b_d) && !extract_json_number(content, "gauge_bytes", b_d)) {
    err_msg = "missing or non-finite 'bytes' / 'gauge_bytes' in " + json_path;
    return false;
  }
  if (b_d <= 0.0 || std::fabs(b_d - std::llround(b_d)) > 1e-9) {
    err_msg = "invalid byte count in " + json_path;
    return false;
  }
  ft.expected_bytes = (size_t)std::llround(b_d);

  if (!extract_json_string(content, "sha256", ft.expected_sha256) &&
      !extract_json_string(content, "gauge_sha256", ft.expected_sha256)) {
    err_msg = "missing 'sha256' / 'gauge_sha256' in " + json_path;
    return false;
  }
  if (!is_valid_sha256_hex(ft.expected_sha256)) {
    err_msg = "invalid 64-hex SHA256 digest in " + json_path;
    return false;
  }

  // Require ALL scientific observable metadata fields to be present and finite
  if (!extract_json_number(content, "plaquette_total", ft.expected_plaq_total)) {
    err_msg = "missing or non-finite 'plaquette_total' in " + json_path;
    return false;
  }
  if (!extract_json_number(content, "plaquette_spatial", ft.expected_plaq_spatial)) {
    err_msg = "missing or non-finite 'plaquette_spatial' in " + json_path;
    return false;
  }
  if (!extract_json_number(content, "plaquette_temporal", ft.expected_plaq_temporal)) {
    err_msg = "missing or non-finite 'plaquette_temporal' in " + json_path;
    return false;
  }
  if (!extract_json_number(content, "ploop_re", ft.expected_ploop_re)) {
    err_msg = "missing or non-finite 'ploop_re' in " + json_path;
    return false;
  }
  if (!extract_json_number(content, "ploop_im", ft.expected_ploop_im)) {
    err_msg = "missing or non-finite 'ploop_im' in " + json_path;
    return false;
  }
  if (!extract_json_number(content, "qcharge", ft.expected_qcharge)) {
    err_msg = "missing or non-finite 'qcharge' in " + json_path;
    return false;
  }
  if (!extract_json_number(content, "energy_total", ft.expected_energy_total)) {
    err_msg = "missing or non-finite 'energy_total' in " + json_path;
    return false;
  }
  if (!extract_json_number(content, "su3_u_err", ft.expected_su3_u_err)) {
    err_msg = "missing or non-finite 'su3_u_err' in " + json_path;
    return false;
  }
  if (!extract_json_number(content, "su3_det_err", ft.expected_su3_det_err)) {
    err_msg = "missing or non-finite 'su3_det_err' in " + json_path;
    return false;
  }

  // Require explicit gauge storage and fermion boundary convention metadata
  std::string gauge_order_str, t_boundary_str, t_phase_slice_str, ploop_conv_str;
  if (!extract_json_string(content, "gauge_order", gauge_order_str) || gauge_order_str != "QUDA_QDP_GAUGE_ORDER") {
    err_msg = "missing or incompatible 'gauge_order' (expected QUDA_QDP_GAUGE_ORDER) in " + json_path;
    return false;
  }
  if (!extract_json_string(content, "fermion_t_boundary", t_boundary_str) || t_boundary_str != "QUDA_ANTI_PERIODIC_T") {
    err_msg = "missing or incompatible 'fermion_t_boundary' (expected QUDA_ANTI_PERIODIC_T) in " + json_path;
    return false;
  }
  if (!extract_json_string(content, "temporal_boundary_phase_slice", t_phase_slice_str) ||
      t_phase_slice_str != "dir=3,t=Lt-1,phase=-1") {
    err_msg = "missing or incompatible 'temporal_boundary_phase_slice' (expected dir=3,t=Lt-1,phase=-1) in " + json_path;
    return false;
  }
  if (!extract_json_string(content, "ploop_convention", ploop_conv_str) ||
      ploop_conv_str != "QUDA_RAW_STORED_TEMPORAL_LINKS_INCLUDES_APBC_SIGN") {
    err_msg = "missing or incompatible 'ploop_convention' in " + json_path;
    return false;
  }
  double dummy_v = 0.0;
  std::string dummy_s;
  if (!extract_json_number(content, "beta", dummy_v) ||
      !extract_json_number(content, "kappa", dummy_v) ||
      !extract_json_number(content, "tau", dummy_v) ||
      !extract_json_number(content, "n_steps", dummy_v) ||
      !extract_json_number(content, "solver_tol", dummy_v) ||
      !extract_json_number(content, "solver_maxiter", dummy_v) ||
      !extract_json_number(content, "seed_base", dummy_v) ||
      !extract_json_string(content, "integrator", dummy_s)) {
    err_msg = "missing or non-finite action/solver/seed metadata in " + json_path;
    return false;
  }

  std::string gauge_file;
  if (extract_json_string(content, "gauge_file", gauge_file) && !gauge_file.empty()) {
    size_t sl = json_path.find_last_of('/');
    std::string dir = (sl != std::string::npos) ? json_path.substr(0, sl) : ".";
    ft.bin_path = (gauge_file[0] == '/') ? gauge_file : (dir + "/" + gauge_file);
  } else if (json_path.size() > 5 && json_path.substr(json_path.size() - 5) == ".json") {
    ft.bin_path = json_path.substr(0, json_path.size() - 5) + ".bin";
  }
  if (ft.bin_path.empty()) {
    err_msg = "cannot resolve binary gauge path for " + json_path;
    return false;
  }
  return true;
}

static bool verify_target_cpu_side(const FieldTarget &ft, size_t gauge_bytes, size_t V_sites,
                                   std::vector<char> &gauge_buf,
                                   std::string &out_sha, double &out_u_err, double &out_det_err,
                                   std::string &err_msg) {
  if (ft.expected_bytes != gauge_bytes) {
    std::ostringstream oss;
    oss << "metadata bytes " << ft.expected_bytes << " != geometry bytes " << gauge_bytes;
    err_msg = oss.str();
    return false;
  }
  struct stat st;
  if (stat(ft.bin_path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || (size_t)st.st_size != gauge_bytes) {
    std::ostringstream oss;
    oss << "file size " << (stat(ft.bin_path.c_str(), &st) == 0 ? (long long)st.st_size : -1LL)
        << " != expected " << gauge_bytes << " bytes";
    err_msg = oss.str();
    return false;
  }

  FILE *f = fopen(ft.bin_path.c_str(), "rb");
  if (!f) {
    err_msg = "could not open binary file " + ft.bin_path;
    return false;
  }
  size_t read_bytes = fread(gauge_buf.data(), 1, gauge_bytes, f);
  fclose(f);
  if (read_bytes != gauge_bytes) {
    err_msg = "short read on " + ft.bin_path;
    return false;
  }

  out_sha = compute_sha256_buffer(gauge_buf.data(), gauge_bytes);
  if (out_sha != ft.expected_sha256) {
    err_msg = "SHA256 mismatch: computed " + out_sha + " != expected " + ft.expected_sha256;
    return false;
  }

  check_su3_host_readonly(reinterpret_cast<const double *>(gauge_buf.data()), ft.X, out_u_err, out_det_err);
  if (!std::isfinite(out_u_err) || !std::isfinite(out_det_err) || out_u_err > 1e-12 || out_det_err > 1e-12) {
    std::ostringstream oss;
    oss << "Read-only SU(3) check failed: max|U^dag U - I|=" << out_u_err << ", max|det(U)-1|=" << out_det_err;
    err_msg = oss.str();
    return false;
  }
  if (std::fabs(out_u_err - ft.expected_su3_u_err) > 1e-12 ||
      std::fabs(out_det_err - ft.expected_su3_det_err) > 1e-12) {
    std::ostringstream oss;
    oss << "SU(3) error metadata mismatch: computed=(" << out_u_err << "," << out_det_err
        << ") vs metadata=(" << ft.expected_su3_u_err << "," << ft.expected_su3_det_err << ")";
    err_msg = oss.str();
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  std::vector<FieldTarget> targets;
  std::string out_tsv = "";
  bool cpu_only = false;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if ((a == "--json" || a == "--manifest") && i + 1 < argc) {
      FieldTarget ft;
      std::string jp = argv[++i];
      std::string err_msg;
      if (!load_field_target_from_json(jp, ft, err_msg)) {
        fprintf(stderr, "FAIL: Metadata JSON validation failed for '%s': %s\n", jp.c_str(), err_msg.c_str());
        return 1;
      }
      targets.push_back(ft);
    } else if (a == "--out-tsv" && i + 1 < argc) {
      out_tsv = argv[++i];
    } else if (a == "--cpu-only") {
      cpu_only = true;
    } else {
      fprintf(stderr, "Usage: %s --json <conf_or_manifest.json> [--json <conf2.json> ...] [--out-tsv <file.tsv>] [--cpu-only]\n", argv[0]);
      return 1;
    }
  }

  if (targets.empty()) {
    fprintf(stderr, "Error: No --json targets provided.\n");
    return 1;
  }

  int X = targets[0].X[0], Y = targets[0].X[1], Z = targets[0].X[2], T = targets[0].X[3];
  for (const auto &t : targets) {
    if (t.X[0] != X || t.X[1] != Y || t.X[2] != Z || t.X[3] != T) {
      fprintf(stderr, "Error: Geometry mismatch across batch targets (%dx%dx%dx%d vs %dx%dx%dx%d)\n",
              X, Y, Z, T, t.X[0], t.X[1], t.X[2], t.X[3]);
      return 1;
    }
  }

  size_t V_sites = (size_t)X * (size_t)Y * (size_t)Z * (size_t)T;
  size_t gauge_bytes = 4 * V_sites * 18 * sizeof(double);
  std::vector<char> gauge_buf(gauge_bytes);

  // First run strict CPU-side validation (file size, SHA-256, finite link entries, SU(3) unitarity/det) on ALL targets
  // BEFORE initializing QUDA/GPU so malformed files fail fast on CPU!
  for (size_t idx = 0; idx < targets.size(); idx++) {
    std::string sha, err_msg;
    double u_err = 0.0, det_err = 0.0;
    if (!verify_target_cpu_side(targets[idx], gauge_bytes, V_sites, gauge_buf, sha, u_err, det_err, err_msg)) {
      fprintf(stderr, "FAIL [%s] (CPU pre-check): %s\n", targets[idx].bin_path.c_str(), err_msg.c_str());
      return 1;
    }
    if (cpu_only) {
      printf("CPU-ONLY PASS [%s]: bytes=%zu, sha256=%s, u_err=%.3e, det_err=%.3e\n",
             targets[idx].bin_path.c_str(), gauge_bytes, sha.substr(0, 12).c_str(), u_err, det_err);
    }
  }

  if (cpu_only) {
    printf("SUCCESS: All %zu gauge field(s) passed CPU-only metadata, SHA-256, and read-only SU(3) checks.\n", targets.size());
    return 0;
  }

  cpu_prec = QUDA_DOUBLE_PRECISION;
  host_gauge_data_type_size = sizeof(double);
  host_spinor_data_type_size = sizeof(double);
  host_clover_data_type_size = sizeof(double);
  prec = QUDA_DOUBLE_PRECISION;
  prec_sloppy = QUDA_DOUBLE_PRECISION;
  prec_precondition = QUDA_DOUBLE_PRECISION;
  prec_eigensolver = QUDA_DOUBLE_PRECISION;
  prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
  link_recon = QUDA_RECONSTRUCT_NO;
  link_recon_sloppy = QUDA_RECONSTRUCT_NO;
  link_recon_precondition = QUDA_RECONSTRUCT_NO;
  link_recon_eigensolver = QUDA_RECONSTRUCT_NO;
  setQudaPrecisions();

  if (!getenv("QUDA_RESOURCE_PATH")) {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd))) {
      std::string tune_dir = std::string(cwd) + "/work/tunecache";
      if (dir_exists(std::string(cwd) + "/../work")) {
        tune_dir = std::string(cwd) + "/../work/tunecache";
      }
      mkdir(tune_dir.c_str(), 0777);
      setenv("QUDA_RESOURCE_PATH", tune_dir.c_str(), 0);
    }
  }

  int dummy_argc = 1;
  char prog_name[] = "verify_gauge_reload";
  char *dummy_argv[] = {prog_name, nullptr};
  initComms(dummy_argc, dummy_argv, gridsize_from_cmdline);
  initQuda(0);

  QudaGaugeParam gauge_param = newQudaGaugeParam();
  setWilsonGaugeParam(gauge_param);
  gauge_param.cpu_prec = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec_precondition = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec_eigensolver = QUDA_DOUBLE_PRECISION;
  gauge_param.cuda_prec_refinement_sloppy = QUDA_DOUBLE_PRECISION;
  gauge_param.reconstruct = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_sloppy = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_precondition = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_eigensolver = QUDA_RECONSTRUCT_NO;
  gauge_param.reconstruct_refinement_sloppy = QUDA_RECONSTRUCT_NO;
  gauge_param.X[0] = X;
  gauge_param.X[1] = Y;
  gauge_param.X[2] = Z;
  gauge_param.X[3] = T;
  gauge_param.use_resident_gauge = 0;
  gauge_param.make_resident_gauge = 1;

  setDims(gauge_param.X);

  void *hostGauge[4];
  for (int i = 0; i < 4; i++) {
    hostGauge[i] = gauge_buf.data() + i * V_sites * 18 * sizeof(double);
  }

  FILE *f_tsv = nullptr;
  if (!out_tsv.empty()) {
    bool write_hdr = !file_exists(out_tsv);
    f_tsv = fopen(out_tsv.c_str(), "a");
    if (f_tsv && write_hdr) {
      fprintf(f_tsv, "file\tbytes\tsha256\tsu3_u_err\tsu3_det_err\tplaq_total\tplaq_diff\tploop_re\tploop_im\tqcharge\tenergy_total\tmax_obs_diff\tstatus\n");
    }
  }

  bool all_passed = true;

  for (size_t idx = 0; idx < targets.size(); idx++) {
    const auto &ft = targets[idx];
    printf("=== QUDA Gauge Reload Verifier [%zu/%zu]: %s ===\n", idx + 1, targets.size(), ft.bin_path.c_str());

    std::string sha, err_msg;
    double u_err = 0.0, det_err = 0.0;
    if (!verify_target_cpu_side(ft, gauge_bytes, V_sites, gauge_buf, sha, u_err, det_err, err_msg)) {
      fprintf(stderr, "FAIL [%s]: %s\n", ft.bin_path.c_str(), err_msg.c_str());
      all_passed = false;
      continue;
    }

    loadGaugeQuda(hostGauge, &gauge_param);

    double plaq[3] = {0.0, 0.0, 0.0};
    plaqQuda(plaq);

    QudaGaugeObservableParam obs = newQudaGaugeObservableParam();
    obs.su_project = QUDA_BOOLEAN_FALSE;
    obs.compute_plaquette = QUDA_BOOLEAN_TRUE;
    obs.compute_polyakov_loop = QUDA_BOOLEAN_TRUE;
    obs.compute_qcharge = QUDA_BOOLEAN_TRUE;
    gaugeObservablesQuda(&obs);

    double d_pt = std::fabs(plaq[0] - ft.expected_plaq_total);
    double d_ps = std::fabs(plaq[1] - ft.expected_plaq_spatial);
    double d_ptemp = std::fabs(plaq[2] - ft.expected_plaq_temporal);
    double d_plr = std::fabs(obs.ploop[0] - ft.expected_ploop_re);
    double d_pli = std::fabs(obs.ploop[1] - ft.expected_ploop_im);
    double d_q = std::fabs(obs.qcharge - ft.expected_qcharge);
    double d_en = std::fabs(obs.energy[0] - ft.expected_energy_total);

    double max_obs_diff = std::max({d_pt, d_ps, d_ptemp, d_plr, d_pli, d_q, d_en});
    bool finite_obs = std::isfinite(d_pt) && std::isfinite(d_ps) && std::isfinite(d_ptemp) &&
                      std::isfinite(d_plr) && std::isfinite(d_pli) && std::isfinite(d_q) && std::isfinite(d_en);

    bool ok = finite_obs && (max_obs_diff <= 1e-12);
    if (!ok) {
      fprintf(stderr, "FAIL [%s]: max observable diff %.3e > 1e-12 (plaq_diff=%.3e)\n",
              ft.bin_path.c_str(), max_obs_diff, d_pt);
      all_passed = false;
    } else {
      printf("PASS [%s]: sha256=%s, u_err=%.3e, det_err=%.3e, plaq=%.17g (diff=%.3e), Q=%.12e, max_obs_diff=%.3e\n",
             ft.bin_path.c_str(), sha.substr(0, 12).c_str(), u_err, det_err, plaq[0], d_pt, obs.qcharge, max_obs_diff);
    }

    if (f_tsv) {
      fprintf(f_tsv, "%s\t%zu\t%s\t%.6e\t%.6e\t%.17g\t%.6e\t%.17g\t%.17g\t%.17g\t%.17g\t%.6e\t%s\n",
              ft.bin_path.c_str(), gauge_bytes, sha.c_str(), u_err, det_err,
              plaq[0], d_pt, obs.ploop[0], obs.ploop[1], obs.qcharge, obs.energy[0],
              max_obs_diff, ok ? "PASS" : "FAIL");
      fflush(f_tsv);
    }
  }

  if (f_tsv) fclose(f_tsv);

  freeGaugeQuda();
  endQuda();
  finalizeComms();

  if (!all_passed) return 1;
  printf("SUCCESS: All %zu gauge field(s) verified in fresh process.\n", targets.size());
  return 0;
}
