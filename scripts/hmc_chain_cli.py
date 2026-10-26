#!/usr/bin/env python3
"""
Reusable Markov-Chain HMC CLI for QUDA Two-Flavour Wilson Ensembles.

Provides documented operational subcommands:
  - start     Start a fresh or checkpoint-seeded Markov chain with persisted chain_spec.json,
              explicit initialization provenance, disk budget guard, and checkpoint rotation.
  - status    Inspect live or stopped chain progress, observables, acceptance, sampling cadence,
              and disk budget usage.
  - stop      Request a graceful post-trajectory checkpointed stop via STOP sentinel + SIGTERM
              and wait for clean manifest publication.
  - resume    Pre-validate checkpoint integrity, reload persisted chain parameters/seed/provenance,
              and continue the Markov chain seamlessly.
  - validate  Verify saved checkpoints, configurations, SHA-256 digests, SU(3) unitarity/determinant,
              storage guard, checkpoint rotation, and unbiased Markov proposal sampling cadence.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
EXP_DIR = Path("/home/vjordan/.local/state/experiment-playground/teams/research-lab/experiments/quda-two-flavour-ensemble-20261025")
QUDA_DIR = EXP_DIR / "quda"
DEFAULT_RUNNER = QUDA_DIR / "build" / "production_ensemble_runner"
VERIFIER_PATH = (SCRIPT_DIR / "verify_ensemble_artifacts.py") if (SCRIPT_DIR / "verify_ensemble_artifacts.py").is_file() else (EXP_DIR / "tanya_storage" / "verify_ensemble_artifacts.py")
LOCK_PATH = EXP_DIR / ".gpu-benchmark.lock"

sys.path.insert(0, str(VERIFIER_PATH.parent))
import verify_ensemble_artifacts as verifier  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_runner_env(tuning_dir: Optional[Path] = None) -> Dict[str, str]:
    env = os.environ.copy()
    ld_paths = [
        str(QUDA_DIR / "build" / "lib_fp4_fixed"),
        str(QUDA_DIR / "cache" / "tests"),
    ]
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths + ([existing_ld] if existing_ld else []))
    tdir = tuning_dir or ((SCRIPT_DIR / "tuning_cache") if (SCRIPT_DIR / "tuning_cache").exists() else (EXP_DIR / "tanya_storage" / "tuning_cache"))
    tdir.mkdir(parents=True, exist_ok=True)
    env.setdefault("QUDA_RESOURCE_PATH", str(tdir))
    env.setdefault("QUDA_ENABLE_TUNING", "1")
    env.setdefault("QUDA_ENABLE_GDR", "0")
    env.setdefault("QUDA_ENABLE_DEVICE_MEMORY_POOL", "0")
    env.setdefault("QUDA_ENABLE_MANAGED_MEMORY", "0")
    return env


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def find_out_dir_pid(out_dir: Path) -> Optional[int]:
    pid_file = out_dir / "runner.pid"
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            cmdline_path = Path(f"/proc/{pid}/cmdline")
            if cmdline_path.is_file():
                cmdline = cmdline_path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
                if "production_ensemble_runner" in cmdline:
                    return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pass

    # Fallback scan of /proc for foreground runner operating on out_dir
    target_str = str(out_dir.resolve())
    for proc_entry in Path("/proc").iterdir():
        if not proc_entry.name.isdigit():
            continue
        pid = int(proc_entry.name)
        if pid == os.getpid():
            continue
        try:
            cmdline = (proc_entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
            if "production_ensemble_runner" in cmdline and target_str in cmdline:
                return pid
        except OSError:
            continue
    return None


def seed_from_existing_checkpoint_dir(src_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Copy the active checkpoint generation referenced by src_dir/checkpoint_manifest.json
    into out_dir (using manifest filenames per Director review #224 pt 3).
    """
    src_man_path = src_dir / "checkpoint_manifest.json"
    if not src_man_path.is_file():
        raise FileNotFoundError(f"Source checkpoint manifest not found: {src_man_path}")

    src_man = json.loads(src_man_path.read_text(encoding="utf-8"))
    gauge_file = src_man["gauge_file"]
    meta_file = src_man["metadata_file"]
    ledger_file = src_man["ledger_file"]

    for fname in [gauge_file, meta_file, ledger_file]:
        src_f = src_dir / fname
        if not src_f.is_file():
            raise FileNotFoundError(f"Source manifest references missing file: {src_f}")
        shutil.copy2(src_f, out_dir / fname)

    src_ens_ledger = src_dir / "ensemble_ledger.tsv"
    if src_ens_ledger.is_file():
        shutil.copy2(src_ens_ledger, out_dir / "ensemble_ledger.tsv")
    else:
        shutil.copy2(src_dir / ledger_file, out_dir / "ensemble_ledger.tsv")

    shutil.copy2(src_man_path, out_dir / "checkpoint_manifest.json")
    return {
        "init_kind": "checkpoint_continuation",
        "source_checkpoint_dir": str(src_dir.resolve()),
        "source_generation": src_man.get("generation"),
        "source_total_traj": src_man.get("total_traj"),
        "source_gauge_file": gauge_file,
        "source_gauge_sha256": src_man.get("gauge_sha256"),
        "source_geometry": src_man.get("geometry"),
        "thermalized_at_target_volume": True,
        "thermalized": True,
        " seeded_at_utc": utc_now_iso(),
    }


def fadvise_dontneed_fd(fd: int, length: int = 0) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(fd, 0, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def periodic_tile_qdp_gauge(
    src_path: Path,
    src_dim: list[int],
    dst_dim: list[int],
    dst_path: Path,
) -> str:
    """
    Periodically tile a saved QDP double-precision SU(3) gauge field from src_dim [Lx0, Ly0, Lz0, Lt0]
    to arbitrary dst_dim [Lx1, Ly1, Lz1, Lt1] (supports integer or non-integer 3D/4D periodic wrapping)
    with anti-periodic temporal boundary adjustment on mu=3, streaming in 4 MiB chunks with
    fdatasync + POSIX_FADV_DONTNEED so GB10 unified-memory page cache is never polluted.
    """
    Lx0, Ly0, Lz0, Lt0 = [int(x) for x in src_dim]
    Lx1, Ly1, Lz1, Lt1 = [int(x) for x in dst_dim]
    if Lx0 <= 0 or Ly0 <= 0 or Lz0 <= 0 or Lt0 <= 0 or Lx1 <= 0 or Ly1 <= 0 or Lz1 <= 0 or Lt1 <= 0:
        raise ValueError(f"Invalid dimensions: src_dim={src_dim} -> dst_dim={dst_dim}")

    V0 = Lx0 * Ly0 * Lz0 * Lt0
    V1 = Lx1 * Ly1 * Lz1 * Lt1
    Vh0 = V0 // 2
    Vh1 = V1 // 2
    import struct, math
    LINK_BYTES = 144
    expected_src = V0 * 4 * LINK_BYTES
    actual_src = src_path.stat().st_size
    if actual_src != expected_src:
        raise ValueError(f"Source gauge size mismatch: got {actual_src}, expected {expected_src}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst_path.with_suffix(dst_path.suffix + ".tmp")
    h = hashlib.sha256()

    neg_table = [struct.Struct("<18d").pack(*(-x for x in vals)) for vals in []]  # unused placeholder
    s18d = struct.Struct("<18d")
    CHUNK_LINKS = 32768

    with open(src_path, "rb") as fs, open(tmp_path, "wb") as fd:
        out_fd = fd.fileno()
        written_since_flush = 0
        for mu in range(4):
            src_dir_bytes = fs.read(V0 * LINK_BYTES)
            if len(src_dir_bytes) != V0 * LINK_BYTES:
                raise RuntimeError(f"Short read on direction {mu} from {src_path}")
            src_mv = memoryview(src_dir_bytes)
            for parity_dst in (0, 1):
                buf = bytearray()
                links_in_buf = 0
                for idx_cb_dst in range(Vh1):
                    full_dst = 2 * idx_cb_dst + parity_dst
                    x = full_dst % Lx1
                    t_tmp = full_dst // Lx1
                    y = t_tmp % Ly1
                    t_tmp //= Ly1
                    z = t_tmp % Lz1
                    t = t_tmp // Lz1
                    if ((x + y + z + t) & 1) != parity_dst:
                        x ^= 1
                    x0 = x % Lx0
                    y0 = y % Ly0
                    z0 = z % Lz0
                    t0 = t % Lt0
                    parity_src = (x0 + y0 + z0 + t0) & 1
                    full_src = ((t0 * Lz0 + z0) * Ly0 + y0) * Lx0 + x0
                    idx_cb_src = full_src // 2
                    src_idx = parity_src * Vh0 + idx_cb_src
                    off = src_idx * LINK_BYTES
                    link_bytes = src_mv[off : off + LINK_BYTES]
                    if mu == 3 and ((t0 == (Lt0 - 1)) != (t == (Lt1 - 1))):
                        vals = s18d.unpack(link_bytes)
                        buf.extend(s18d.pack(*(-v for v in vals)))
                    else:
                        buf.extend(link_bytes)
                    links_in_buf += 1
                    if links_in_buf >= CHUNK_LINKS:
                        fd.write(buf)
                        h.update(buf)
                        written_since_flush += len(buf)
                        buf.clear()
                        links_in_buf = 0
                        if written_since_flush >= (64 << 20):
                            fd.flush()
                            os.fdatasync(out_fd)
                            fadvise_dontneed_fd(out_fd, 0)
                            written_since_flush = 0
                if buf:
                    fd.write(buf)
                    h.update(buf)
                    written_since_flush += len(buf)
                    buf.clear()
        fd.flush()
        os.fsync(out_fd)
        fadvise_dontneed_fd(out_fd, 0)
        fadvise_dontneed_fd(fs.fileno(), 0)
    os.replace(tmp_path, dst_path)
    return h.hexdigest()


def tile_from_checkpoint_source(tile_source: Path, dst_dim: List[int], out_dir: Path, src_dim_override: Optional[List[int]] = None) -> Tuple[Path, Dict[str, Any]]:
    tile_source = tile_source.resolve()
    src_man_path: Optional[Path] = None
    src_bin_path: Optional[Path] = None
    src_man: Dict[str, Any] = {}

    if tile_source.is_dir():
        cand = tile_source / "checkpoint_manifest.json"
        if not cand.is_file():
            raise FileNotFoundError(f"No checkpoint_manifest.json in directory {tile_source}")
        src_man_path = cand
        src_man = json.loads(cand.read_text(encoding="utf-8"))
        src_bin_path = tile_source / src_man["gauge_file"]
    elif tile_source.name.endswith(".json"):
        src_man_path = tile_source
        src_man = json.loads(tile_source.read_text(encoding="utf-8"))
        gauge_fname = src_man.get("gauge_file")
        if gauge_fname:
            src_bin_path = tile_source.parent / gauge_fname
        else:
            src_bin_path = tile_source.with_suffix(".bin")
    else:
        src_bin_path = tile_source
        cand_man = tile_source.parent / "checkpoint_manifest.json"
        if cand_man.is_file():
            m = json.loads(cand_man.read_text(encoding="utf-8"))
            if m.get("gauge_file") == tile_source.name:
                src_man_path = cand_man
                src_man = m

    if src_bin_path is None or not src_bin_path.is_file():
        raise FileNotFoundError(f"Resolved source gauge binary does not exist: {src_bin_path}")

    src_dim = list(src_dim_override) if src_dim_override else list(src_man.get("geometry", []))
    if len(src_dim) != 4:
        raise ValueError(f"Cannot determine 4D source geometry for {src_bin_path}; pass --tile-source-dim")

    src_sha = compute_sha256(src_bin_path)
    if src_man.get("gauge_sha256") and src_man["gauge_sha256"] != src_sha:
        raise ValueError(f"Source checkpoint SHA-256 mismatch for {src_bin_path}")

    tiled_bin = out_dir / "init_tiled_gauge.bin"
    tiled_sha = periodic_tile_qdp_gauge(src_bin_path, src_dim, dst_dim, tiled_bin)
    prov = {
        "init_kind": "periodic_tile_from_checkpoint",
        "initialization_kind": "periodic_tile_from_checkpoint",
        "source_checkpoint": str(src_bin_path),
        "source_manifest": str(src_man_path) if src_man_path else "",
        "source_generation": src_man.get("generation"),
        "source_total_traj": src_man.get("total_traj"),
        "source_gauge_file": src_bin_path.name,
        "source_gauge_sha256": src_sha,
        "source_geometry": src_dim,
        "target_geometry": list(dst_dim),
        "tile_factors": [round(dst_dim[i] / src_dim[i], 6) for i in range(4)],
        "tiled_gauge_file": tiled_bin.name,
        "tiled_gauge_sha256": tiled_sha,
        "thermalized": False,
        "thermalized_at_target_volume": False,
        "created_at_utc": utc_now_iso(),
    }
    return tiled_bin, prov


def build_runner_cmd(spec: Dict[str, Any], resume: bool) -> List[str]:
    runner_bin = spec.get("runner_bin", str(DEFAULT_RUNNER))
    dim = spec["geometry"]
    cmd = [
        runner_bin,
        "--mode", "run",
        "--dim", str(dim[0]), str(dim[1]), str(dim[2]), str(dim[3]),
        "--beta", str(spec["beta"]),
        "--kappa", str(spec["kappa"]),
        "--csw", str(spec.get("csw", 0.0)),
        "--tau", str(spec["tau"]),
        "--n-steps", str(spec["nsteps"]),
        "--tol", str(spec["tol"]),
        "--niter", str(spec["maxiter"]),
        "--sloppy-prec", str(spec["sloppy_prec"]),
        "--sloppy-recon", str(spec["sloppy_recon"]),
        "--precise-recon", str(spec.get("precise_recon", spec["sloppy_recon"])),
        "--extended-recon", str(spec.get("extended_recon", spec["sloppy_recon"])),
        "--reliable-delta", str(spec["reliable_delta"]),
        "--pipeline", str(spec.get("pipeline", 0)),
        "--sloppy-accum", str(spec.get("sloppy_accum", 0)),
        "--accum-pipeline", str(spec.get("accum_pipeline", 1)),
        "--chrono-dim", str(spec.get("chrono_dim", 0)),
        "--seed", str(spec["seed"]),
        "--seed-stride", str(spec["seed_stride"]),
        "--warmup", str(spec["warmup"]),
        "--prod", str(spec["prod"]),
        "--stride", str(spec["save_stride"]),
        "--max-proposals", str(spec["max_proposals"]),
        "--keep-checkpoints", str(spec["keep_checkpoints"]),
        "--max-out-dir-bytes", str(spec.get("max_out_dir_bytes", 0)),
        "--min-free-disk-bytes", str(spec.get("min_free_disk_bytes", 107374182400)),
        "--out-dir", str(spec["out_dir"]),
        "--omp-su3", str(spec.get("omp_su3", 1)),
        "--memstream-log", str(spec.get("memstream_log", 1)),
    ]
    if str(spec.get("start_type", "")) in ("cold", "unit"):
        cmd.extend(["--unit-gauge", "true"])
    if not resume and spec.get("init_gauge_bin"):
        cmd.extend(["--init-gauge-bin", str(spec["init_gauge_bin"])])
    if "initialization_provenance" in spec and spec["initialization_provenance"]:
        cmd.extend(["--init-provenance-json", json.dumps(spec["initialization_provenance"], separators=(",", ":"))])
    if spec.get("test_force_reject_trajs"):
        cmd.extend(["--test-force-reject-trajs", str(spec["test_force_reject_trajs"]), "--allow-test-reject"])
    if resume:
        cmd.append("--resume")
    return cmd


def launch_runner(cmd: List[str], out_dir: Path, background: bool, use_gpu_lock: bool) -> int:
    env = build_runner_env()
    log_path = out_dir / "run.log"
    pid_path = out_dir / "runner.pid"
    for sname in ("STOP", "STOP_REQUESTED", ".stop_requested"):
        sp = out_dir / sname
        if sp.exists():
            sp.unlink()

    if background:
        log_f = log_path.open("a", encoding="utf-8")
        log_f.write(f"\n=== [{utc_now_iso()}] Launching background runner: {' '.join(cmd)} ===\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
        print(json.dumps({
            "action": "launched_background",
            "pid": proc.pid,
            "out_dir": str(out_dir),
            "log_path": str(log_path),
            "pid_path": str(pid_path),
        }, indent=2))
        return 0

    lock_f = None
    try:
        if use_gpu_lock:
            LOCK_PATH.touch(exist_ok=True)
            lock_f = LOCK_PATH.open("r+")
            fcntl.flock(lock_f, fcntl.LOCK_EX)

        with log_path.open("a", encoding="utf-8") as log_f:
            log_f.write(f"\n=== [{utc_now_iso()}] Launching foreground runner: {' '.join(cmd)} ===\n")
            log_f.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()
            rc = proc.wait()
            if pid_path.exists():
                try:
                    pid_path.unlink()
                except OSError:
                    pass
            return rc
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f, fcntl.LOCK_UN)
                lock_f.close()
            except OSError:
                pass


def cmd_start(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and (out_dir / "checkpoint_manifest.json").exists() and not args.force_clean:
        sys.stderr.write(
            f"ERROR: {out_dir}/checkpoint_manifest.json already exists. "
            f"Use 'resume' to continue the existing chain or pass --force-clean to overwrite.\n"
        )
        return 1

    if args.force_clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check real filesystem free disk guard before starting
    v_sites = int(args.dim[0]) * int(args.dim[1]) * int(args.dim[2]) * int(args.dim[3])
    gauge_bytes = 4 * v_sites * 18 * 8
    free_disk = verifier.get_fs_free_bytes(out_dir)
    if args.min_free_disk_bytes > 0 and free_disk > 0 and free_disk < (2 * gauge_bytes + args.min_free_disk_bytes):
        sys.stderr.write(
            f"ERROR: Filesystem free disk ({free_disk} bytes) is below required "
            f"transient_write ({2 * gauge_bytes} bytes) + min_free_disk_bytes ({args.min_free_disk_bytes} bytes).\n"
        )
        return 2

    resume_right_away = False
    tiled_bin_path: Optional[Path] = None
    if args.init_provenance_json:
        try:
            init_prov = json.loads(args.init_provenance_json)
        except Exception as exc:
            sys.stderr.write(f"ERROR: Invalid --init-provenance-json: {exc}\n")
            return 1
    elif getattr(args, "tile_from", ""):
        tiled_bin_path, init_prov = tile_from_checkpoint_source(
            Path(args.tile_from),
            list(args.dim),
            out_dir,
            src_dim_override=args.tile_source_dim,
        )
    elif args.init_from_checkpoint_dir:
        src_dir = Path(args.init_from_checkpoint_dir).resolve()
        init_prov = seed_from_existing_checkpoint_dir(src_dir, out_dir)
        resume_right_away = True
    else:
        init_prov = {
            "init_kind": f"fresh_{args.start_type}_start",
            "initialization_kind": f"fresh_{args.start_type}_start",
            "start_type": args.start_type,
            "geometry": list(args.dim),
            "seed_base": args.seed,
            "seed_stride": args.seed_stride,
            "thermalized": False,
            "thermalized_at_target_volume": False,
            "created_at_utc": utc_now_iso(),
        }

    spec: Dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": utc_now_iso(),
        "out_dir": str(out_dir),
        "runner_bin": str(Path(args.runner_bin).resolve()),
        "geometry": list(args.dim),
        "beta": args.beta,
        "kappa": args.kappa,
        "csw": args.csw,
        "tau": args.tau,
        "nsteps": args.nsteps,
        "tol": args.tol,
        "maxiter": args.maxiter,
        "sloppy_prec": args.sloppy_prec,
        "sloppy_recon": args.sloppy_recon,
        "reliable_delta": args.reliable_delta,
        "pipeline": args.pipeline,
        "sloppy_accum": args.sloppy_accum,
        "accum_pipeline": args.accum_pipeline,
        "chrono_dim": args.chrono_dim,
        "seed": args.seed,
        "seed_stride": args.seed_stride,
        "warmup": args.warmup,
        "prod": args.prod,
        "save_stride": args.save_stride,
        "max_proposals": args.max_proposals,
        "keep_checkpoints": args.keep_checkpoints,
        "max_out_dir_bytes": args.max_out_dir_bytes,
        "min_free_disk_bytes": args.min_free_disk_bytes,
        "start_type": args.start_type,
        "omp_su3": args.omp_su3,
        "memstream_log": args.memstream_log,
        "initialization_provenance": init_prov,
    }
    if tiled_bin_path is not None:
        spec["init_gauge_bin"] = str(tiled_bin_path)
    if args.test_force_reject_trajs:
        spec["test_force_reject_trajs"] = args.test_force_reject_trajs

    spec_path = out_dir / "chain_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    cmd = build_runner_cmd(spec, resume=resume_right_away)
    return launch_runner(cmd, out_dir, background=args.background, use_gpu_lock=args.gpu_lock)


def cmd_resume(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    man_path = out_dir / "checkpoint_manifest.json"
    spec_path = out_dir / "chain_spec.json"

    if not man_path.is_file():
        sys.stderr.write(f"ERROR: Cannot resume; {man_path} does not exist.\n")
        return 1

    active_pid = find_out_dir_pid(out_dir)
    if active_pid is not None:
        sys.stderr.write(f"ERROR: Runner process PID {active_pid} is still active for {out_dir}.\n")
        return 1

    # Pre-validate checkpoint integrity before resuming
    if not args.skip_pre_validate:
        rep = verifier.verify_out_dir(
            out_dir,
            tol_su3=args.tol_su3,
            allow_checkpoint_only=args.allow_checkpoint_only,
        )
        if not rep["pass"]:
            sys.stderr.write(
                f"ERROR: Pre-resume checkpoint validation failed for {out_dir}:\n"
                + json.dumps(rep["errors"], indent=2)
                + "\n"
            )
            return 1

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if spec_path.is_file():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    else:
        # Reconstruct chain_spec.json from checkpoint_manifest.json
        spec = {
            "schema_version": 1,
            "created_at_utc": utc_now_iso(),
            "out_dir": str(out_dir),
            "runner_bin": str(DEFAULT_RUNNER),
            "geometry": manifest["geometry"],
            "beta": manifest["beta"],
            "kappa": manifest["kappa"],
            "csw": manifest.get("csw", 0.0),
            "tau": manifest["tau"],
            "nsteps": manifest["n_steps"],
            "tol": manifest["solver_tol"],
            "maxiter": manifest["solver_maxiter"],
            "sloppy_prec": manifest.get("sloppy_prec", "half"),
            "sloppy_recon": manifest.get("sloppy_recon", 12),
            "reliable_delta": manifest.get("reliable_delta", 0.1),
            "pipeline": manifest.get("pipeline", 0),
            "sloppy_accum": manifest.get("sloppy_accum", 0),
            "accum_pipeline": manifest.get("accum_pipeline", 1),
            "chrono_dim": manifest.get("chrono_dim", 0),
            "seed": manifest["seed_base"],
            "seed_stride": manifest["seed_stride"],
            "warmup": manifest["warmup_done"],
            "prod": max(manifest["saved_confs"] + 1, 10),
            "save_stride": 1,
            "max_proposals": 1000000,
            "keep_checkpoints": 2,
            "max_out_dir_bytes": 0,
            "min_free_disk_bytes": 107374182400,
            "start_type": "cold",
            "omp_su3": 1,
            "memstream_log": 1,
            "initialization_provenance": manifest.get("initialization_provenance", {}),
        }

    # Apply operational target overrides (without altering physics or seed)
    if args.warmup is not None:
        spec["warmup"] = args.warmup
    if args.prod is not None:
        spec["prod"] = args.prod
    if args.save_stride is not None:
        spec["save_stride"] = args.save_stride
    if args.max_proposals is not None:
        spec["max_proposals"] = args.max_proposals
    if args.keep_checkpoints is not None:
        spec["keep_checkpoints"] = args.keep_checkpoints
    if args.max_out_dir_bytes is not None:
        spec["max_out_dir_bytes"] = args.max_out_dir_bytes
    if args.min_free_disk_bytes is not None:
        spec["min_free_disk_bytes"] = args.min_free_disk_bytes
    if args.reliable_delta is not None:
        spec["reliable_delta"] = args.reliable_delta
    if args.maxiter is not None:
        spec["maxiter"] = args.maxiter
    if args.test_force_reject_trajs is not None:
        if args.test_force_reject_trajs == "":
            spec.pop("test_force_reject_trajs", None)
        else:
            spec["test_force_reject_trajs"] = args.test_force_reject_trajs

    spec["last_resumed_at_utc"] = utc_now_iso()
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    cmd = build_runner_cmd(spec, resume=True)
    return launch_runner(cmd, out_dir, background=args.background, use_gpu_lock=args.gpu_lock)


def cmd_stop(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    if not out_dir.is_dir():
        sys.stderr.write(f"ERROR: Directory does not exist: {out_dir}\n")
        return 1

    stop_msg = f"stop requested at {utc_now_iso()}\n"
    stop_sentinel = out_dir / "STOP"
    for sname in ("STOP", "STOP_REQUESTED", ".stop_requested"):
        (out_dir / sname).write_text(stop_msg, encoding="utf-8")
    pid = find_out_dir_pid(out_dir)

    if pid is not None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pid = None

    t0 = time.time()
    while pid is not None and (time.time() - t0) < args.wait_sec:
        try:
            os.kill(pid, 0)
            time.sleep(0.5)
        except ProcessLookupError:
            pid = None
            break

    still_running = pid is not None
    pid_path = out_dir / "runner.pid"
    if not still_running and pid_path.exists():
        try:
            pid_path.unlink()
        except OSError:
            pass

    man_path = out_dir / "checkpoint_manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    res = {
        "action": "graceful_stop",
        "out_dir": str(out_dir),
        "stop_sentinel_written": str(stop_sentinel),
        "signaled_pid": pid,
        "stopped_cleanly": not still_running,
        "waited_sec": round(time.time() - t0, 2),
        "generation": manifest.get("generation"),
        "total_traj": manifest.get("total_traj"),
        "saved_confs": manifest.get("saved_confs"),
        "gauge_file": manifest.get("gauge_file"),
        "gauge_sha256": manifest.get("gauge_sha256"),
    }
    print(json.dumps(res, indent=2))
    return 0 if not still_running else 1


def collect_status(out_dir: Path) -> Dict[str, Any]:
    out_dir = out_dir.resolve()
    pid = find_out_dir_pid(out_dir)
    man_path = out_dir / "checkpoint_manifest.json"
    spec_path = out_dir / "chain_spec.json"
    ledger_path = out_dir / "ensemble_ledger.tsv"

    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    spec = json.loads(spec_path.read_text(encoding="utf-8")) if spec_path.is_file() else {}

    physical_bytes = verifier.sum_regular_bytes(out_dir) if out_dir.is_dir() else 0
    max_bytes = int(spec.get("max_out_dir_bytes", 0))
    min_free_bytes = int(spec.get("min_free_disk_bytes", 107374182400))
    free_disk_bytes = verifier.get_fs_free_bytes(out_dir)
    mem_metrics = verifier.get_host_memory_metrics(out_dir, pid=pid)
    ckpt_bins = sorted(p.name for p in out_dir.glob("ckpt_gen_*.bin")) if out_dir.is_dir() else []
    conf_bins = sorted(p.name for p in out_dir.glob("conf.*.bin")) if out_dir.is_dir() else []

    ledger_summary: Dict[str, Any] = {}
    if ledger_path.is_file():
        lines = [ln for ln in ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) > 1 and lines[0].startswith("traj\t"):
            headers = lines[0].split("\t")
            rows = [dict(zip(headers, ln.split("\t"))) for ln in lines[1:]]
            acc_cnt = sum(1 for r in rows if r.get("accepted") == "1")
            prod_rows = [r for r in rows if r.get("phase") == "prod"]
            prod_acc_cnt = sum(1 for r in prod_rows if r.get("accepted") == "1")
            mean_sec = (
                sum(float(r.get("traj_sec", 0.0)) for r in rows) / len(rows)
                if rows else 0.0
            )
            max_res = max((float(r.get("max_true_res", 0.0)) for r in rows), default=0.0)
            ledger_summary = {
                "total_rows": len(rows),
                "overall_accepted": acc_cnt,
                "overall_acceptance_rate": round(acc_cnt / len(rows), 4) if rows else 0.0,
                "prod_proposals": len(prod_rows),
                "prod_accepted": prod_acc_cnt,
                "prod_acceptance_rate": round(prod_acc_cnt / len(prod_rows), 4) if prod_rows else 0.0,
                "mean_traj_sec": round(mean_sec, 4),
                "max_true_res": max_res,
            }

    return {
        "out_dir": str(out_dir),
        "state": "RUNNING" if pid is not None else ("STOPPED" if man_path.is_file() else "UNINITIALIZED"),
        "active_pid": pid,
        "stop_sentinel_present": any((out_dir / s).is_file() for s in ("STOP", "STOP_REQUESTED", ".stop_requested")),
        "geometry": manifest.get("geometry") or spec.get("geometry"),
        "generation": manifest.get("generation"),
        "total_traj": manifest.get("total_traj"),
        "warmup_done": manifest.get("warmup_done"),
        " warmup_target": spec.get("warmup"),
        "prod_proposals_done": manifest.get("prod_proposals_done"),
        "accepted_prod_traj": manifest.get("accepted_prod_traj"),
        "saved_confs": manifest.get("saved_confs"),
        "prod_target_confs": spec.get("prod"),
        "save_stride": spec.get("save_stride"),
        "state_id": manifest.get("state_id"),
        "multiplicity": manifest.get("multiplicity"),
        "latest_observables": {
            "plaquette_total": manifest.get("plaquette_total"),
            "qcharge": manifest.get("qcharge"),
            "energy_total": manifest.get("energy_total"),
            "su3_u_err": manifest.get("su3_u_err"),
            "su3_det_err": manifest.get("su3_det_err"),
        } if manifest else None,
        "storage": {
            "physical_regular_bytes": physical_bytes,
            "physical_gib": round(physical_bytes / (1024 ** 3), 4),
            "max_out_dir_bytes": max_bytes,
            "budget_used_pct": round(100.0 * physical_bytes / max_bytes, 2) if max_bytes > 0 else 0.0,
            "free_disk_bytes": free_disk_bytes,
            "free_disk_gib": round(free_disk_bytes / (1024 ** 3), 4),
            "min_free_disk_bytes": min_free_bytes,
            "free_disk_guard_pass": (min_free_bytes <= 0 or free_disk_bytes >= min_free_bytes),
            "keep_checkpoints": spec.get("keep_checkpoints"),
            "retained_checkpoints": ckpt_bins,
            "saved_conf_files_count": len(conf_bins),
        },
        "memory": mem_metrics,
        "ledger_summary": ledger_summary,
        "initialization_provenance": manifest.get("initialization_provenance") or spec.get("initialization_provenance"),
        "active_checkpoint": {
            "gauge_file": manifest.get("gauge_file"),
            "gauge_sha256": manifest.get("gauge_sha256"),
            "last_seed_traj": manifest.get("last_seed_traj"),
        } if manifest else None,
    }


def cmd_status(args: argparse.Namespace) -> int:
    st = collect_status(Path(args.out_dir))
    if args.json:
        print(json.dumps(st, indent=2))
        return 0

    print(f"=== QUDA Markov-Chain Status: {st['out_dir']} ===")
    print(f"State              : {st['state']}" + (f" (PID {st['active_pid']})" if st["active_pid"] else ""))
    print(f"Geometry           : {st['geometry']}")
    print(
        f"Progress           : gen={st['generation']} | total_traj={st['total_traj']} | "
        f"warmup={st['warmup_done']}/{st[' warmup_target']} | "
        f"prod_proposals={st['prod_proposals_done']} | accepted_prod={st['accepted_prod_traj']} | "
        f"saved_confs={st['saved_confs']}/{st['prod_target_confs']}"
    )
    print(f"Markov State       : state_id={st['state_id']} | multiplicity={st['multiplicity']}")
    if st["latest_observables"]:
        obs = st["latest_observables"]
        print(
            f"Latest Observables : plaq={obs['plaquette_total']} | Q={obs['qcharge']} | "
            f"u_err={obs['su3_u_err']} | det_err={obs['su3_det_err']}"
        )
    if st["ledger_summary"]:
        ls = st["ledger_summary"]
        print(
            f"Acceptance & Speed : overall={ls['overall_accepted']}/{ls['total_rows']} ({ls['overall_acceptance_rate']*100:.1f}%) | "
            f"prod={ls['prod_accepted']}/{ls['prod_proposals']} ({ls['prod_acceptance_rate']*100:.1f}%) | "
            f"mean_traj={ls['mean_traj_sec']}s | max_true_res={ls['max_true_res']:.3e}"
        )
    stor = st["storage"]
    print(
        f"Storage & Rotation : {stor['physical_regular_bytes']} B ({stor['physical_gib']} GiB, "
        f"free_disk={stor['free_disk_gib']} GiB, min_free_guard={stor['min_free_disk_bytes']} B) | "
        f"retained_ckpts={stor['retained_checkpoints']} | saved_confs={stor['saved_conf_files_count']}"
    )
    mem = st.get("memory") or {}
    if mem:
        print(
            f"Unified Memory     : MemAvailable={mem.get('mem_available_gib')} GiB "
            f"(min={mem.get('mem_available_min_gib')} GiB) | VmHWM={mem.get('vm_hwm_kb')} kB | VmRSS={mem.get('vm_rss_kb')} kB"
        )
    if st["initialization_provenance"]:
        print(f"Provenance         : {json.dumps(st['initialization_provenance'])}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    rep = verifier.verify_out_dir(
        out_dir,
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quda_ensemble_cli.py",
        description="Reusable Markov-chain management CLI (start | status | stop | resume | validate) for QUDA two-flavour Wilson HMC ensembles.",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    # start
    ps = sub.add_parser("start", help="Start a fresh, checkpoint-seeded, or periodic-tiled HMC Markov chain")
    ps.add_argument("--out-dir", required=True, help="Output directory for checkpoints, configurations, and ledger")
    ps.add_argument("--dim", nargs=4, type=int, default=[16, 16, 16, 32], help="Lattice geometry Lx Ly Lz Lt")
    ps.add_argument("--beta", type=float, default=5.6, help="Wilson gauge coupling beta")
    ps.add_argument("--kappa", type=float, default=0.156, help="Wilson hopping parameter kappa")
    ps.add_argument("--csw", type=float, default=0.0, help="Clover c_sw coefficient (0.0 for unimproved Wilson)")
    ps.add_argument("--tau", type=float, default=1.0, help="MD trajectory length tau")
    ps.add_argument("--nsteps", "--n-steps", dest="nsteps", type=int, default=25, help="Omelyan integrator steps per trajectory")
    ps.add_argument("--tol", type=float, default=1e-8, help="Exact FP64 solver residual tolerance")
    ps.add_argument("--maxiter", type=int, default=2000, help="Maximum CG iterations")
    ps.add_argument("--sloppy-prec", default="half", choices=["double", "single", "half", "quarter"], help="Sloppy precision")
    ps.add_argument("--sloppy-recon", type=int, default=12, choices=[8, 12, 18], help="Sloppy link reconstruction")
    ps.add_argument("--reliable-delta", type=float, default=0.1, help="Reliable update delta")
    ps.add_argument("--pipeline", type=int, default=0, help="CG pipeline length")
    ps.add_argument("--sloppy-accum", type=int, default=0, help="Sloppy accumulator flag")
    ps.add_argument("--accum-pipeline", type=int, default=1, help="Accumulator pipeline")
    ps.add_argument("--chrono-dim", type=int, default=0, help="Chronological forecast dimension (0=disabled)")
    ps.add_argument("--seed", "--seed-base", dest="seed", type=int, default=20261025, help="Deterministic base RNG seed")
    ps.add_argument("--seed-stride", type=int, default=100000, help="Per-trajectory seed stride")
    ps.add_argument("--warmup", type=int, default=0, help="Number of warmup trajectories")
    ps.add_argument("--prod", type=int, default=10, help="Target number of saved production configurations")
    ps.add_argument("--save-stride", type=int, default=1, help="Save configuration every N production proposals")
    ps.add_argument("--max-proposals", type=int, default=1000000, help="Max HMC proposals in this invocation")
    ps.add_argument("--keep-checkpoints", type=int, default=2, help="Number of rolling checkpoint generations to retain (>= 1)")
    ps.add_argument("--max-out-dir-bytes", type=int, default=0, help="Hard pre-write storage budget in bytes (default 0 = unlimited)")
    ps.add_argument("--min-free-disk-bytes", type=int, default=107374182400, help="Real filesystem free-disk guard via statvfs in bytes (default 100 GiB)")
    ps.add_argument("--start-type", default="cold", choices=["cold", "hot", "unit"], help="Initial gauge configuration if starting fresh")
    ps.add_argument("--tile-from", default="", help="Periodically tile initial gauge from a smaller-volume checkpoint directory, checkpoint_manifest.json, or ckpt_gen_*.bin")
    ps.add_argument("--tile-source-dim", nargs=4, type=int, default=None, help="Optional explicit 4D geometry of --tile-from source if no manifest is present")
    ps.add_argument("--init-from-checkpoint-dir", default="", help="Seed from an existing validated checkpoint directory using manifest filenames")
    ps.add_argument("--init-provenance-json", default="", help="Explicit JSON object recording initialization provenance")
    ps.add_argument("--omp-su3", type=int, default=1, help="OpenMP threads for host SU(3) check")
    ps.add_argument("--memstream-log", type=int, default=1, help="Use in-memory POSIX stream for solver log capture")
    ps.add_argument("--runner-bin", default=str(DEFAULT_RUNNER), help="Path to production_ensemble_runner binary")
    ps.add_argument("--test-force-reject-trajs", default="", help="Comma-separated 1-based trajectories to force Metropolis rejection (TEST-ONLY)")
    ps.add_argument("--background", action="store_true", help="Run in background daemon mode and write runner.pid")
    ps.add_argument("--gpu-lock", action="store_true", help="Acquire .gpu-benchmark.lock during foreground run")
    ps.add_argument("--force-clean", action="store_true", help="Remove existing --out-dir before starting fresh")
    ps.set_defaults(func=cmd_start)

    # status
    pst = sub.add_parser("status", help="Show Markov-chain progress, observables, sampling cadence, and storage status")
    pst.add_argument("--out-dir", required=True, help="Ensemble output directory")
    pst.add_argument("--json", action="store_true", help="Emit status as machine-readable JSON")
    pst.set_defaults(func=cmd_status)

    # stop
    psp = sub.add_parser("stop", help="Gracefully stop a running Markov chain after current trajectory checkpoint")
    psp.add_argument("--out-dir", required=True, help="Ensemble output directory")
    psp.add_argument("--wait-sec", "--timeout", dest="wait_sec", type=float, default=120.0, help="Max seconds to wait for current trajectory to finish")
    psp.add_argument("--wait", action="store_true", help="Wait for running trajectory to checkpoint and exit cleanly")
    psp.set_defaults(func=cmd_stop)

    # resume
    pr = sub.add_parser("resume", help="Pre-validate checkpoint and resume Markov chain preserving parameters, RNG seed, and provenance")
    pr.add_argument("--out-dir", required=True, help="Ensemble output directory")
    pr.add_argument("--warmup", type=int, default=None, help="Override target warmup trajectories")
    pr.add_argument("--prod", type=int, default=None, help="Override target saved production configurations")
    pr.add_argument("--save-stride", type=int, default=None, help="Override proposal save stride")
    pr.add_argument("--max-proposals", type=int, default=None, help="Override max proposals for this run")
    pr.add_argument("--keep-checkpoints", type=int, default=None, help="Override retained checkpoints count K")
    pr.add_argument("--max-out-dir-bytes", type=int, default=None, help="Override hard storage cap in bytes (0 = unlimited)")
    pr.add_argument("--min-free-disk-bytes", type=int, default=None, help="Override minimum free disk bytes guard")
    pr.add_argument("--reliable-delta", type=float, default=None, help="Override reliable update delta")
    pr.add_argument("--maxiter", type=int, default=None, help="Override solver maxiter")
    pr.add_argument("--test-force-reject-trajs", default=None, help="Override test forced-rejection trajectory list")
    pr.add_argument("--tol-su3", type=float, default=1e-12, help="SU(3) tolerance for pre-resume validation")
    pr.add_argument("--allow-checkpoint-only", action="store_true", help="Allow resuming from a checkpoint-only seeded directory")
    pr.add_argument("--skip-pre-validate", action="store_true", help="Skip Python pre-validation before launching runner --resume")
    pr.add_argument("--background", action="store_true", help="Run resumed chain in background daemon mode")
    pr.add_argument("--gpu-lock", action="store_true", help="Acquire .gpu-benchmark.lock during foreground run")
    pr.set_defaults(func=cmd_resume)

    # validate / verify
    for cmd_name in ["validate", "verify"]:
        pv = sub.add_parser(cmd_name, help="Validate checkpoints, saved configurations, SU(3) errors, storage budget, and proposal sampling")
        pv.add_argument("--out-dir", required=True, help="Ensemble output directory")
        pv.add_argument("--tol-su3", type=float, default=1e-12, help="Max allowed SU(3) unitarity/det error")
        pv.add_argument("--expect-keep-checkpoints", type=int, default=0, help="Expected retained checkpoint count K")
        pv.add_argument("--max-out-dir-bytes", type=int, default=0, help="Max allowed directory bytes (default 0 = unlimited)")
        pv.add_argument("--min-free-disk-bytes", type=int, default=107374182400, help="Minimum required free filesystem bytes via statvfs (default 100 GiB)")
        pv.add_argument("--allow-checkpoint-only", action="store_true", help="Allow checkpoint-only directory without prior conf.*.bin")
        pv.add_argument("--json-out", default="", help="Optional path to write JSON validation report")
        pv.set_defaults(func=cmd_validate)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
