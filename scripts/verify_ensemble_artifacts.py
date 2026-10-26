#!/usr/bin/env python3
"""
Standalone verifier for QUDA two-flavour HMC ensemble output directories.
Zero external Python dependencies (uses Python stdlib + auto-compiled OpenMP C++ ctypes kernel
with pure-Python struct fallback).
Validates:
  - checkpoint_manifest.json & ensemble_ledger.tsv consistency and SHA-256 hashes
  - All ckpt_gen_*.{bin,meta.json,ledger.tsv} files present in --out-dir
  - All conf.*.{bin,json} files present in --out-dir
  - Exact byte sizes (4 * V_sites * 18 * 8 bytes)
  - SHA-256 digests against JSON metadata and trajectory ledger rows
  - Finite FP64 entries across all gauge links
  - SU(3) unitarity error max|U^dag U - I| <= 1e-12 and determinant error max|det(U) - 1| <= 1e-12
    after removing the antiperiodic temporal boundary phase (dir=3, t=Lt-1, phase=-1)
    in parity-ordered QUDA_QDP_GAUGE_ORDER layout.
"""

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_C_KERNEL_SRC = r"""
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>

extern "C" int check_su3_file_qdp_apbc(
    const char *filepath,
    int Lx, int Ly, int Lz, int Lt,
    double *out_u_err,
    double *out_det_err,
    long *out_nonfinite
) {
  size_t V_sites = (size_t)Lx * (size_t)Ly * (size_t)Lz * (size_t)Lt;
  size_t Vh_sites = V_sites / 2;
  size_t lower_bound = ((size_t)(Lx / 2)) * (size_t)Ly * (size_t)Lz * (size_t)(Lt - 1);
  size_t expected_doubles = 4 * V_sites * 18;

  FILE *f = std::fopen(filepath, "rb");
  if (!f) return -1;
  double *buf = new (std::nothrow) double[expected_doubles];
  if (!buf) {
    std::fclose(f);
    return -2;
  }
  size_t nread = std::fread(buf, sizeof(double), expected_doubles, f);
#if defined(_POSIX_ADVISORY_INFO) && (_POSIX_ADVISORY_INFO > 0)
  (void)posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
#endif
  std::fclose(f);
  if (nread != expected_doubles) {
    delete[] buf;
    return -3;
  }

  using cd = std::complex<double>;
  double u_err_max = 0.0;
  double det_err_max = 0.0;
  long nonfinite_count = 0;

  #pragma omp parallel for collapse(3) reduction(max:u_err_max,det_err_max) reduction(+:nonfinite_count)
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

  delete[] buf;
  *out_u_err = u_err_max;
  *out_det_err = det_err_max;
  *out_nonfinite = nonfinite_count;
  return 0;
}
"""

_C_LIB = None


def _get_su3_clib():
    global _C_LIB
    if _C_LIB is not None:
        return _C_LIB
    so_dir = Path(__file__).resolve().parent
    so_path = so_dir / f".libsu3_verify_py{os.getuid()}.so"
    cpp_path = so_dir / f".libsu3_verify_py{os.getuid()}.cpp"
    if not so_path.is_file():
        try:
            cpp_path.write_text(_C_KERNEL_SRC, encoding="utf-8")
            subprocess.run(
                ["g++", "-O3", "-fopenmp", "-shared", "-fPIC", str(cpp_path), "-o", str(so_path)],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except Exception:
            _C_LIB = False
            return _C_LIB
    try:
        lib = ctypes.CDLL(str(so_path))
        lib.check_su3_file_qdp_apbc.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_long),
        ]
        lib.check_su3_file_qdp_apbc.restype = ctypes.c_int
        _C_LIB = lib
    except Exception:
        _C_LIB = False
    return _C_LIB


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return h.hexdigest()


def get_fs_free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    try:
        st = os.statvfs(str(target))
        return int(st.f_bavail) * int(st.f_frsize)
    except OSError:
        return 0


def get_host_memory_metrics(out_dir: Path | None = None, pid: int | None = None) -> Dict[str, Any]:
    meminfo: Dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            parts = line.split(":")
            if len(parts) == 2:
                k = parts[0].strip()
                v_parts = parts[1].strip().split()
                if v_parts and v_parts[0].isdigit():
                    meminfo[k] = int(v_parts[0]) * 1024  # bytes
    except OSError:
        pass

    vm_hwm_kb = 0
    vm_rss_kb = 0
    target_pid = pid
    if target_pid is None and out_dir is not None:
        pid_file = out_dir / "runner.pid"
        if pid_file.is_file():
            try:
                target_pid = int(pid_file.read_text(encoding="utf-8").strip())
            except ValueError:
                target_pid = None
    if target_pid is None:
        target_pid = os.getpid()

    try:
        status_path = Path(f"/proc/{target_pid}/status")
        if status_path.is_file():
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmHWM:"):
                    vals = [int(x) for x in line.split() if x.isdigit()]
                    if vals:
                        vm_hwm_kb = vals[0]
                elif line.startswith("VmRSS:"):
                    vals = [int(x) for x in line.split() if x.isdigit()]
                    if vals:
                        vm_rss_kb = vals[0]
    except OSError:
        pass

    mem_avail_now = meminfo.get("MemAvailable", 0)
    mem_avail_min = mem_avail_now

    # Also inspect run.log or telemetry if present in out_dir for historical VmHWM / MemAvailable
    if out_dir is not None and (out_dir / "run.log").is_file():
        try:
            log_txt = (out_dir / "run.log").read_text(encoding="utf-8", errors="ignore")
            for m in re.finditer(r"VmHWM[=:\s]+(\d+)\s*kB", log_txt, re.IGNORECASE):
                vm_hwm_kb = max(vm_hwm_kb, int(m.group(1)))
            for m in re.finditer(r"MemAvailable[=:\s]+(\d+)\s*kB", log_txt, re.IGNORECASE):
                val_b = int(m.group(1)) * 1024
                if val_b > 0 and (mem_avail_min == 0 or val_b < mem_avail_min):
                    mem_avail_min = val_b
        except OSError:
            pass

    return {
        "mem_total_bytes": meminfo.get("MemTotal", 0),
        "mem_free_bytes": meminfo.get("MemFree", 0),
        "mem_available_bytes": mem_avail_now,
        "mem_available_min_bytes": mem_avail_min,
        "mem_available_gib": round(mem_avail_now / (1024 ** 3), 4),
        "mem_available_min_gib": round(mem_avail_min / (1024 ** 3), 4),
        "cached_bytes": meminfo.get("Cached", 0),
        "dirty_bytes": meminfo.get("Dirty", 0),
        "vm_hwm_kb": vm_hwm_kb,
        "vm_rss_kb": vm_rss_kb,
    }


def check_su3_qdp_apbc(bin_path: Path, geometry: List[int]) -> Tuple[bool, float, float, int]:
    """
    Replicates check_su3_host_readonly() from production_ensemble_runner.cpp:
      - Layout: QUDA_QDP_GAUGE_ORDER: [dir=0..3][parity=0..1][j=0..Vh_sites-1][row=0..2][col=0..2][re,im]
      - APBC sign: for dir == 3 and j >= (Lx // 2) * Ly * Lz * (Lt - 1), U_raw = -U_SU3 (phase = -1.0).
      - Unitarity error: max_{links, i, k_col} |(U^dag U)_{i, k_col} - delta_{i, k_col}|
      - Determinant error: max_{links} |det(U) - 1|
    Returns: (all_finite, max_u_err, max_det_err, nonfinite_count)
    """
    lx, ly, lz, lt = [int(x) for x in geometry]
    lib = _get_su3_clib()
    if lib:
        u_err = ctypes.c_double(0.0)
        det_err = ctypes.c_double(0.0)
        nonfinite = ctypes.c_long(0)
        rc = lib.check_su3_file_qdp_apbc(
            str(bin_path).encode("utf-8"),
            lx,
            ly,
            lz,
            lt,
            ctypes.byref(u_err),
            ctypes.byref(det_err),
            ctypes.byref(nonfinite),
        )
        if rc == 0:
            nf = int(nonfinite.value)
            if nf > 0:
                return False, float("nan"), float("nan"), nf
            return True, float(u_err.value), float(det_err.value), 0

    # Pure-Python fallback
    v_sites = lx * ly * lz * lt
    vh_sites = v_sites // 2
    lower_bound = (lx // 2) * ly * lz * (lt - 1)
    data = bin_path.read_bytes()
    u_err_max = 0.0
    det_err_max = 0.0
    nonfinite_cnt = 0
    unpack_link = struct.Struct("<18d").unpack_from

    for d in range(4):
        for parity in range(2):
            for j in range(vh_sites):
                m = d * v_sites + parity * vh_sites + j
                vals = unpack_link(data, m * 144)
                if not all(math.isfinite(v) for v in vals):
                    nonfinite_cnt += 1
                    continue
                s = -1.0 if (d == 3 and j >= lower_bound) else 1.0
                U = [
                    [complex(s * vals[0], s * vals[1]), complex(s * vals[2], s * vals[3]), complex(s * vals[4], s * vals[5])],
                    [complex(s * vals[6], s * vals[7]), complex(s * vals[8], s * vals[9]), complex(s * vals[10], s * vals[11])],
                    [complex(s * vals[12], s * vals[13]), complex(s * vals[14], s * vals[15]), complex(s * vals[16], s * vals[17])],
                ]
                for i in range(3):
                    for k_col in range(3):
                        acc = U[0][i].conjugate() * U[0][k_col] + U[1][i].conjugate() * U[1][k_col] + U[2][i].conjugate() * U[2][k_col]
                        target = 1.0 + 0.0j if i == k_col else 0.0j
                        err = abs(acc - target)
                        if err > u_err_max:
                            u_err_max = err
                det = (
                    U[0][0] * (U[1][1] * U[2][2] - U[1][2] * U[2][1])
                    - U[0][1] * (U[1][0] * U[2][2] - U[1][2] * U[2][0])
                    + U[0][2] * (U[1][0] * U[2][1] - U[1][1] * U[2][0])
                )
                derr = abs(det - (1.0 + 0.0j))
                if derr > det_err_max:
                    det_err_max = derr

    if nonfinite_cnt > 0:
        return False, float("nan"), float("nan"), nonfinite_cnt
    return True, u_err_max, det_err_max, 0


def sum_regular_bytes(out_dir: Path) -> int:
    total = 0
    seen_inodes = set()
    for root, _dirs, files in os.walk(out_dir, followlinks=False):
        for fname in files:
            fpath = Path(root) / fname
            try:
                st = fpath.lstat()
                if os.path.isfile(fpath) and not os.path.islink(fpath):
                    ino_key = (st.st_dev, st.st_ino)
                    if ino_key not in seen_inodes:
                        seen_inodes.add(ino_key)
                        total += st.st_size
            except OSError:
                pass
    return total


def verify_out_dir(
    out_dir: Path,
    tol_su3: float = 1e-12,
    expect_keep_checkpoints: int = 0,
    max_out_dir_bytes: int = 0,
    min_free_disk_bytes: int = 107374182400,
    allow_checkpoint_only: bool = False,
) -> Dict[str, Any]:
    errors: List[str] = []
    free_disk_bytes = get_fs_free_bytes(out_dir)
    mem_metrics = get_host_memory_metrics(out_dir)
    report: Dict[str, Any] = {
        "out_dir": str(out_dir),
        "pass": False,
        "regular_bytes": 0,
        "max_out_dir_bytes": max_out_dir_bytes,
        "free_disk_bytes": free_disk_bytes,
        "free_disk_gib": round(free_disk_bytes / (1024 ** 3), 4),
        "min_free_disk_bytes": min_free_disk_bytes,
        "free_disk_guard_pass": (min_free_disk_bytes <= 0 or free_disk_bytes >= min_free_disk_bytes),
        "memory": mem_metrics,
        "manifest": None,
        "ledger": None,
        "sampling": None,
        "checkpoints": [],
        "configurations": [],
        "errors": errors,
    }

    if not out_dir.is_dir():
        errors.append(f"Output directory does not exist: {out_dir}")
        return report

    regular_bytes = sum_regular_bytes(out_dir)
    report["regular_bytes"] = regular_bytes
    if max_out_dir_bytes > 0 and regular_bytes > max_out_dir_bytes:
        errors.append(
            f"Directory regular_bytes ({regular_bytes}) exceeds max_out_dir_bytes ({max_out_dir_bytes})"
        )
    if min_free_disk_bytes > 0 and free_disk_bytes > 0 and free_disk_bytes < min_free_disk_bytes:
        errors.append(
            f"Filesystem free_disk_bytes ({free_disk_bytes}) is below min_free_disk_bytes ({min_free_disk_bytes})"
        )

    # Check for leftover transient .tmp files
    tmp_files = [p.name for p in out_dir.iterdir() if ".tmp." in p.name]
    if tmp_files:
        errors.append(f"Found leftover transient .tmp files in out_dir: {sorted(tmp_files)}")

    man_path = out_dir / "checkpoint_manifest.json"
    if not man_path.is_file():
        errors.append(f"Missing checkpoint_manifest.json in {out_dir}")
        return report

    try:
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"Failed to parse checkpoint_manifest.json: {exc}")
        return report

    report["manifest"] = {
        "generation": manifest.get("generation"),
        "total_traj": manifest.get("total_traj"),
        "warmup_done": manifest.get("warmup_done"),
        "prod_proposals_done": manifest.get("prod_proposals_done", manifest.get("prod_traj")),
        "accepted_prod_traj": manifest.get("accepted_prod_traj"),
        "saved_confs": manifest.get("saved_confs"),
        "state_id": manifest.get("state_id"),
        "multiplicity": manifest.get("multiplicity"),
        "geometry": manifest.get("geometry"),
        "gauge_file": manifest.get("gauge_file"),
        "gauge_sha256": manifest.get("gauge_sha256"),
        "gauge_bytes": manifest.get("gauge_bytes"),
        "initialization_provenance": manifest.get("initialization_provenance"),
    }

    geometry = manifest.get("geometry")
    if not (isinstance(geometry, list) and len(geometry) == 4 and all(isinstance(x, int) and x > 0 for x in geometry)):
        errors.append(f"Invalid geometry in checkpoint_manifest.json: {geometry}")
        return report

    v_sites = geometry[0] * geometry[1] * geometry[2] * geometry[3]
    expected_gauge_bytes = 4 * v_sites * 18 * 8
    if manifest.get("gauge_bytes") != expected_gauge_bytes:
        errors.append(
            f"Manifest gauge_bytes ({manifest.get('gauge_bytes')}) != expected ({expected_gauge_bytes})"
        )

    if manifest.get("gauge_order") != "QUDA_QDP_GAUGE_ORDER":
        errors.append(f"Invalid gauge_order in manifest: {manifest.get('gauge_order')}")
    if manifest.get("fermion_t_boundary") != "QUDA_ANTI_PERIODIC_T":
        errors.append(f"Invalid fermion_t_boundary in manifest: {manifest.get('fermion_t_boundary')}")
    if manifest.get("temporal_boundary_phase_slice") != "dir=3,t=Lt-1,phase=-1":
        errors.append(
            f"Invalid temporal_boundary_phase_slice in manifest: {manifest.get('temporal_boundary_phase_slice')}"
        )

    for obs_key in [
        "plaquette_total",
        "plaquette_spatial",
        "plaquette_temporal",
        "ploop_re",
        "ploop_im",
        "qcharge",
        "energy_total",
        "su3_u_err",
        "su3_det_err",
    ]:
        val = manifest.get(obs_key)
        if not (isinstance(val, (int, float)) and math.isfinite(float(val))):
            errors.append(f"Manifest observable '{obs_key}' is missing or non-finite: {val}")

    # Verify ensemble_ledger.tsv and manifest ledger_file
    ledger_path = out_dir / "ensemble_ledger.tsv"
    if not ledger_path.is_file():
        errors.append("Missing ensemble_ledger.tsv")
        ledger_rows_by_traj: Dict[int, Dict[str, str]] = {}
    else:
        ens_ledger_sha = compute_sha256(ledger_path)
        if ens_ledger_sha != manifest.get("ledger_sha256"):
            errors.append(
                f"ensemble_ledger.tsv SHA-256 ({ens_ledger_sha}) != manifest ledger_sha256 ({manifest.get('ledger_sha256')})"
            )
        lines = [ln for ln in ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines or not lines[0].startswith("traj\t"):
            errors.append("ensemble_ledger.tsv missing header row")
            ledger_rows_by_traj = {}
        else:
            headers = lines[0].split("\t")
            data_lines = lines[1:]
            ledger_rows_by_traj = {}
            for idx, ln in enumerate(data_lines, start=1):
                cols = ln.split("\t")
                if len(cols) != len(headers):
                    errors.append(f"Ledger row {idx} column count {len(cols)} != header count {len(headers)}")
                    continue
                row_dict = dict(zip(headers, cols))
                t_num = int(row_dict["traj"])
                if t_num != idx:
                    errors.append(f"Ledger row {idx} has non-sequential traj={t_num}")
                ledger_rows_by_traj[t_num] = row_dict

            total_traj = int(manifest.get("total_traj", -1))
            if len(data_lines) != total_traj or manifest.get("ledger_rows") != total_traj:
                errors.append(
                    f"Ledger row count mismatch: file={len(data_lines)}, manifest.ledger_rows={manifest.get('ledger_rows')}, manifest.total_traj={total_traj}"
                )
            if total_traj in ledger_rows_by_traj:
                last_row = ledger_rows_by_traj[total_traj]
                if last_row.get("state_sha256") != manifest.get("gauge_sha256"):
                    errors.append(
                        f"Final ledger row state_sha256 ({last_row.get('state_sha256')}) != manifest gauge_sha256 ({manifest.get('gauge_sha256')})"
                    )
            report["ledger"] = {
                "rows": len(data_lines),
                "sha256": ens_ledger_sha,
            }

    # Discover and validate all ckpt_gen_*.bin files
    ckpt_bins = sorted(out_dir.glob("ckpt_gen_*.bin"))
    ckpt_metas = sorted(out_dir.glob("ckpt_gen_*.meta.json"))
    ckpt_ledgers = sorted(out_dir.glob("ckpt_gen_*.ledger.tsv"))

    bin_gens = [p.name[: len("ckpt_gen_00000")] for p in ckpt_bins]
    meta_gens = [p.name[: len("ckpt_gen_00000")] for p in ckpt_metas]
    ledger_gens = [p.name[: len("ckpt_gen_00000")] for p in ckpt_ledgers]
    if bin_gens != meta_gens or bin_gens != ledger_gens:
        errors.append(
            f"Mismatched ckpt_gen_* triplets: bins={bin_gens}, metas={meta_gens}, ledgers={ledger_gens}"
        )

    active_ckpt_bin = out_dir / str(manifest.get("gauge_file", ""))
    if not active_ckpt_bin.is_file():
        errors.append(f"Active manifest gauge_file missing on disk: {active_ckpt_bin.name}")

    if expect_keep_checkpoints > 0:
        if len(ckpt_bins) > expect_keep_checkpoints:
            errors.append(
                f"Found {len(ckpt_bins)} checkpoint generations on disk, expected <= {expect_keep_checkpoints}"
            )
        cur_gen = int(manifest.get("generation", 0))
        expected_min_gen = max(1, cur_gen - expect_keep_checkpoints + 1)
        expected_names = [f"ckpt_gen_{g:05d}.bin" for g in range(expected_min_gen, cur_gen + 1)]
        actual_names = [p.name for p in ckpt_bins]
        if actual_names != expected_names:
            errors.append(
                f"Retained checkpoint generations {actual_names} != expected {expected_names} for K={expect_keep_checkpoints}"
            )

    for cbin in ckpt_bins:
        m_match = re.match(r"^ckpt_gen_(\d+)\.bin$", cbin.name)
        if not m_match:
            errors.append(f"Unexpected checkpoint filename format: {cbin.name}")
            continue
        gen_num = int(m_match.group(1))
        cmeta = out_dir / f"ckpt_gen_{gen_num:05d}.meta.json"
        cledg = out_dir / f"ckpt_gen_{gen_num:05d}.ledger.tsv"

        c_entry: Dict[str, Any] = {"file": cbin.name, "generation": gen_num, "pass": False}
        if not cmeta.is_file() or not cledg.is_file():
            errors.append(f"Missing meta.json or ledger.tsv for {cbin.name}")
            report["checkpoints"].append(c_entry)
            continue

        meta_obj = json.loads(cmeta.read_text(encoding="utf-8"))
        actual_bytes = cbin.stat().st_size
        actual_sha = compute_sha256(cbin)
        ledg_sha = compute_sha256(cledg)

        # Notice in ckpt_gen_%05d.meta.json the keys are "bytes" and "sha256"
        meta_bytes = meta_obj.get("bytes", meta_obj.get("gauge_bytes"))
        meta_sha256 = meta_obj.get("sha256", meta_obj.get("gauge_sha256"))

        if actual_bytes != expected_gauge_bytes or meta_bytes != expected_gauge_bytes:
            errors.append(
                f"{cbin.name}: byte size mismatch (disk={actual_bytes}, meta={meta_bytes}, expected={expected_gauge_bytes})"
            )
        if actual_sha != meta_sha256:
            errors.append(
                f"{cbin.name}: SHA-256 mismatch (disk={actual_sha} vs meta={meta_sha256})"
            )

        if gen_num == manifest.get("generation"):
            meta_file_sha = compute_sha256(cmeta)
            if meta_file_sha != manifest.get("metadata_sha256"):
                errors.append(
                    f"{cmeta.name}: SHA-256 ({meta_file_sha}) != manifest metadata_sha256 ({manifest.get('metadata_sha256')})"
                )
            if actual_sha != manifest.get("gauge_sha256"):
                errors.append(
                    f"{cbin.name}: SHA-256 ({actual_sha}) != manifest gauge_sha256 ({manifest.get('gauge_sha256')})"
                )
            if ledg_sha != manifest.get("ledger_sha256"):
                errors.append(
                    f"{cledg.name}: SHA-256 ({ledg_sha}) != manifest ledger_sha256 ({manifest.get('ledger_sha256')})"
                )

        finite_ok, u_err, det_err, nonfinite_cnt = check_su3_qdp_apbc(cbin, geometry)
        if not finite_ok or nonfinite_cnt > 0:
            errors.append(f"{cbin.name}: found non-finite FP64 entries (count={nonfinite_cnt})")
        if not (math.isfinite(u_err) and u_err <= tol_su3):
            errors.append(f"{cbin.name}: SU(3) unitarity error {u_err:.3e} exceeds {tol_su3:.3e}")
        if not (math.isfinite(det_err) and det_err <= tol_su3):
            errors.append(f"{cbin.name}: SU(3) determinant error {det_err:.3e} exceeds {tol_su3:.3e}")

        c_entry.update(
            {
                "bytes": actual_bytes,
                "sha256": actual_sha,
                "finite": finite_ok,
                "su3_u_err": u_err,
                "su3_det_err": det_err,
                "meta_su3_u_err": meta_obj.get("su3_u_err"),
                "meta_su3_det_err": meta_obj.get("su3_det_err"),
                "pass": (
                    actual_bytes == expected_gauge_bytes
                    and actual_sha == meta_sha256
                    and finite_ok
                    and u_err <= tol_su3
                    and det_err <= tol_su3
                ),
            }
        )
        report["checkpoints"].append(c_entry)

    # Discover and validate all conf.*.bin files
    conf_bins = sorted(out_dir.glob("conf.*.bin"))
    saved_confs = int(manifest.get("saved_confs", 0))
    if not allow_checkpoint_only and len(conf_bins) != saved_confs:
        errors.append(
            f"Configuration count mismatch: found {len(conf_bins)} conf.*.bin files on disk, manifest saved_confs={saved_confs}"
        )

    inode_check_cache: Dict[Tuple[int, int], Tuple[str, bool, float, float, int]] = {}
    unique_state_ids = set()
    rejected_repeat_confs = 0

    first_conf_num: Optional[int] = None
    for idx, fbin in enumerate(conf_bins, start=1):
        m_conf = re.match(r"^conf\.(\d+)\.bin$", fbin.name)
        if not m_conf:
            errors.append(f"Unexpected configuration filename: {fbin.name}")
            continue
        conf_num = int(m_conf.group(1))
        if first_conf_num is None:
            first_conf_num = conf_num
        expected_conf_num = (first_conf_num + idx - 1) if allow_checkpoint_only else idx
        if conf_num != expected_conf_num:
            errors.append(f"Non-contiguous configuration index: expected {expected_conf_num}, got {conf_num} ({fbin.name})")
        fjson = out_dir / f"conf.{conf_num:05d}.json"
        f_entry: Dict[str, Any] = {"file": fbin.name, "configuration": conf_num, "pass": False}
        if not fjson.is_file():
            errors.append(f"Missing metadata JSON {fjson.name} for {fbin.name}")
            report["configurations"].append(f_entry)
            continue

        cmeta = json.loads(fjson.read_text(encoding="utf-8"))
        st = fbin.lstat()
        ino_key = (st.st_dev, st.st_ino)
        actual_bytes = st.st_size

        if ino_key in inode_check_cache:
            actual_sha, finite_ok, u_err, det_err, nonfinite_cnt = inode_check_cache[ino_key]
        else:
            actual_sha = compute_sha256(fbin)
            finite_ok, u_err, det_err, nonfinite_cnt = check_su3_qdp_apbc(fbin, geometry)
            inode_check_cache[ino_key] = (actual_sha, finite_ok, u_err, det_err, nonfinite_cnt)

        if actual_bytes != expected_gauge_bytes or cmeta.get("bytes") != expected_gauge_bytes:
            errors.append(
                f"{fbin.name}: byte size mismatch (disk={actual_bytes}, json={cmeta.get('bytes')}, expected={expected_gauge_bytes})"
            )
        if actual_sha != cmeta.get("sha256"):
            errors.append(
                f"{fbin.name}: SHA-256 mismatch (disk={actual_sha} vs json={cmeta.get('sha256')})"
            )

        traj_num = int(cmeta.get("trajectory", -1))
        conf_state_id = cmeta.get("state_id")
        conf_mult = cmeta.get("multiplicity")
        unique_state_ids.add(conf_state_id)
        if isinstance(conf_mult, int) and conf_mult > 1:
            rejected_repeat_confs += 1

        if "prod_proposal" in cmeta and "save_stride" in cmeta:
            prod_prop = int(cmeta["prod_proposal"])
            s_stride = int(cmeta["save_stride"])
            expected_prod_prop = conf_num * s_stride
            if prod_prop != expected_prod_prop:
                errors.append(
                    f"{fbin.name}: prod_proposal ({prod_prop}) != configuration ({conf_num}) * save_stride ({s_stride})"
                )

        if traj_num in ledger_rows_by_traj:
            lrow = ledger_rows_by_traj[traj_num]
            if lrow.get("state_sha256") != actual_sha:
                errors.append(
                    f"{fbin.name}: SHA-256 ({actual_sha}) != ledger traj {traj_num} state_sha256 ({lrow.get('state_sha256')})"
                )
            if str(conf_state_id) != lrow.get("state_id"):
                errors.append(
                    f"{fbin.name}: state_id ({conf_state_id}) != ledger traj {traj_num} state_id ({lrow.get('state_id')})"
                )
            if str(conf_mult) != lrow.get("multiplicity"):
                errors.append(
                    f"{fbin.name}: multiplicity ({conf_mult}) != ledger traj {traj_num} multiplicity ({lrow.get('multiplicity')})"
                )
            if "accepted_at_proposal" in cmeta:
                expected_acc = lrow.get("accepted") == "1"
                if bool(cmeta["accepted_at_proposal"]) != expected_acc:
                    errors.append(
                        f"{fbin.name}: accepted_at_proposal ({cmeta['accepted_at_proposal']}) != ledger accepted ({lrow.get('accepted')})"
                    )
        else:
            errors.append(f"{fbin.name}: trajectory {traj_num} not found in ensemble_ledger.tsv")

        if not finite_ok or nonfinite_cnt > 0:
            errors.append(f"{fbin.name}: found non-finite FP64 entries (count={nonfinite_cnt})")
        if not (math.isfinite(u_err) and u_err <= tol_su3):
            errors.append(f"{fbin.name}: SU(3) unitarity error {u_err:.3e} exceeds {tol_su3:.3e}")
        if not (math.isfinite(det_err) and det_err <= tol_su3):
            errors.append(f"{fbin.name}: SU(3) determinant error {det_err:.3e} exceeds {tol_su3:.3e}")

        f_entry.update(
            {
                "trajectory": traj_num,
                "prod_proposal": cmeta.get("prod_proposal", cmeta.get("prod_traj")),
                "state_id": conf_state_id,
                "multiplicity": conf_mult,
                "accepted_at_proposal": cmeta.get("accepted_at_proposal", cmeta.get("accepted")),
                "hardlink_nlink": st.st_nlink,
                "bytes": actual_bytes,
                "sha256": actual_sha,
                "finite": finite_ok,
                "su3_u_err": u_err,
                "su3_det_err": det_err,
                "json_su3_u_err": cmeta.get("su3_u_err"),
                "json_su3_det_err": cmeta.get("su3_det_err"),
                "pass": (
                    actual_bytes == expected_gauge_bytes
                    and actual_sha == cmeta.get("sha256")
                    and finite_ok
                    and u_err <= tol_su3
                    and det_err <= tol_su3
                ),
            }
        )
        report["configurations"].append(f_entry)

    report["sampling"] = {
        "saved_confs": len(conf_bins),
        "unique_states_in_saved_confs": len(unique_state_ids),
        "unique_conf_inodes": len(inode_check_cache),
        "rejected_repeat_confs": rejected_repeat_confs,
    }

    report["pass"] = len(errors) == 0
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify QUDA HMC ensemble artifacts in --out-dir")
    parser.add_argument("--out-dir", required=True, help="Ensemble output directory to verify")
    parser.add_argument("--tol-su3", type=float, default=1e-12, help="Max allowed SU(3) unitarity/det error")
    parser.add_argument(
        "--expect-keep-checkpoints",
        type=int,
        default=0,
        help="If > 0, verify retained ckpt_gen_* count and generation numbers match K",
    )
    parser.add_argument(
        "--max-out-dir-bytes",
        type=int,
        default=0,
        help="If > 0, verify total regular file bytes in --out-dir <= limit (default 0 = unlimited)",
    )
    parser.add_argument(
        "--min-free-disk-bytes",
        type=int,
        default=107374182400,
        help="Minimum required free filesystem bytes via statvfs (default 100 GiB; 0 disables)",
    )
    parser.add_argument(
        "--allow-checkpoint-only",
        action="store_true",
        help="Allow checkpoint-only seed directories where historical conf.*.bin files were not copied",
    )
    parser.add_argument("--json-out", default="", help="Optional path to write full JSON verification report")
    args = parser.parse_args()

    rep = verify_out_dir(
        Path(args.out_dir),
        tol_su3=args.tol_su3,
        expect_keep_checkpoints=args.expect_keep_checkpoints,
        max_out_dir_bytes=args.max_out_dir_bytes,
        min_free_disk_bytes=args.min_free_disk_bytes,
        allow_checkpoint_only=args.allow_checkpoint_only,
    )
    out_text = json.dumps(rep, indent=2)
    print(out_text)
    if args.json_out:
        Path(args.json_out).write_text(out_text + "\n", encoding="utf-8")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
