# Maximum-Volume Two-Flavour Wilson-Clover HMC on DGX Spark GB10 (`56^3 x 96` & `56^3 x 112`) and Large-Lattice FP4 Runbook

## 1. Summary of Validated Maximum-Volume Ladder (`quda-max-volume-20261026`)
On NVIDIA DGX Spark (GB10 Grace Blackwell Superchip, `sm_121`, 48 SMs, 119.7 GiB unified LPDDR5x + 16.0 GiB swap), we eliminated host/device memory overheads (`3.08 GiB` per 1M sites removed via immediate `host_rand` release, device pool flushing around `gaugeObservables`, `posix_fadvise(POSIX_FADV_DONTNEED)`, and periodic 4D gauge tiling) and executed two-flavour ($N_f=2$) dynamical Wilson-Clover HMC (`QUDA_RECONSTRUCT_NO`, `QUDA_DOUBLE_PRECISION` outer / `QUDA_SINGLE_PRECISION` inner):

| Lattice Volume | Sites ($V$) | Binary Checkpoint Size | Peak Shared UMA | Min `MemAvailable` | Peak System Swap | Trajectory 1 Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`48^3 x 96`** | `10,616,832` | `6.115 GiB` (`6,115,295,232 B`) | `64.62 GiB` | `55.08 GiB` | `0.00 GiB` | `accepted=1`, `dH=-0.04540858`, `P=0.565345979207405` |
| **`56^3 x 96`** | `16,859,136` | `9.711 GiB` (`9,710,862,336 B`) | `102.31 GiB` | `17.39 GiB` | `0.625 GiB` | **Max Swap-Free In-Core**: Traj 1 (`dH=+0.05633622`) & Traj 2 (`dH=-0.27885850`) accepted |
| **`56^3 x 112`** | `19,668,992` | `10.551 GiB` (`11,329,339,392 B`) | `118.501 GiB` | `1.176 GiB` | `16.000 GiB` | **Completed under substantial UMA pressure & swap**: `accepted=1`, `dH=-6.5701730251312256`, `P=0.527532560885565`, `Q=243.290575935150`, `8372.93s` (`139.55 min`) |

## 2. Precision Provenance Erratum (`#302`)
- In the original `56^3 x 112` trajectory 1 run (`PID 1341480`), `dH = -6.5701730251312256` and `dH_per_site = -3.3403709885749232e-07` were captured directly as the IEEE-754 `binary64` return value of `hmcTrajectoryQuda()`.
- However, `H_init` and `H_final` (`430294800`) were parsed from `libquda.so` stdout formatted with `%e` (`4.302948e+08`, 7 significant digits) because `PID 1341480` launched before `libquda.so` was rebuilt with `%.17g`.
- Commits `c21aa274b` and `347743e2c` fix both the origin format string (`%.17g` in `lib/interface_quda.cpp`) and `production_ensemble_runner.cpp` provenance validation (`H_precision_provenance` only emits `"binary64_17sig"` when parsed `H_init`/`H_final` tokens carry >7 significant digits and match `dH`, otherwise emitting `"decimal_7sig_quda_log_e_format"` alongside `"dH_precision_provenance": "binary64_17sig"`).
- Legacy binary/JSON receipts are preserved unmodified with sidecar errata (`conf.00001.erratum.json` and `H_precision_provenance_erratum.json`).

## 3. CLI Runbook (`scripts/quda_ensemble_cli.py`)
```bash
# 1. Start a new ensemble (or tile an equilibrated seed configuration)
python3 scripts/quda_ensemble_cli.py start --ensemble-dir /path/to/ens --dims 56,56,56,96 --n-traj 10

# 2. Inspect ensemble status, ledger, and checkpoint provenance
python3 scripts/quda_ensemble_cli.py status --ensemble-dir /path/to/ens

# 3. Resume from last verified checkpoint generation
python3 scripts/quda_ensemble_cli.py resume --ensemble-dir /path/to/ens --n-traj 10

# 4. Run full SHA-256 checkpoint + CPU observable validation
python3 scripts/quda_ensemble_cli.py validate --ensemble-dir /path/to/ens
```
