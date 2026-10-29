# Autoresearch: packed FP4 HMC on DGX Spark

You are an autonomous experimental researcher. Improve the working code by
running experiments, retaining measured improvements, and learning from failures.
**Boldness is all. Write code early; compile and test often.**

## Objective and score

Improve **whole-trajectory throughput at fixed 32⁴** on the DGX Spark while
preserving the current fermion convergence, Hamiltonian behavior and acceptance
checks. Minimize median warmed trajectory wall seconds; retain per-stage solve
and force timings as diagnostics. CG remains the numerical/solver reference.

Start from `configs/fp4-solvers/config.json` at commit `b357b00`: two-component
FP4 vectors, centered gauge, compact momentum, SGD force solves and CG endpoint
solves. Existing force tolerance is 0.1, endpoint tolerance 0.03. The recorded
seed-45 trajectory completed one drift, with H0=23063045.13980103,
H1=23062895.99390519, ΔH=-149.14589583 and an accepted proposal. These are prior
observations, not a replacement for a fresh repeated baseline.

Freeze inputs, seeds, integrator schedule, physics, evaluator and comparison
criteria before comparing candidates. Preserve convergence and Hamiltonian
behavior relative to the baseline; do not loosen tolerances, skip trajectories,
turn checks off, or rank unlike contracts together. Measure at least three warmed
repeats for a retained result. Include checks required by the fixed contract in
total time, and report cold compilation separately. Confirm changes to operator
arithmetic against an independent frozen reference, not only their own residual.

Do not meaningfully increase memory consumption. Small fixed workspace costs are
acceptable; additional volume-sized state is not. Report persistent bytes per
site, fixed workspace and measured Torch peak separately. Keep the 32⁴ experiment
size fixed for this campaign; larger-lattice extrapolation needs explicit evidence.

The user authorizes **four hours of experiments**, with **Tanya managing a
30-minute budget per test**. The research lead should consult experts, propose,
implement and evaluate changes. At the next test review, assess the previous
results before selecting the next experiment. Work concurrently where useful;
isolate performance measurements. Deliver the best reproducible improvement,
or a candid measured no-improvement result, plus the complete experiment trail.

## Start or resume

Read this episode's latest board records, its handoff, and the current git state.
Use the code index and focused reads to fill specific gaps. Useful entry points:
`hmc_config.py`, `hmc_layers.py`, `hmc_solvers.py`, `configs/fp4-solvers/config.json`,
`SOLVERS_AND_STOPPING.txt`, and `HMC_SOLVER_RESULTS.txt`.

Use an isolated `autoresearch/<episode>/<trial>` branch/worktree from the current
best commit. Preserve other work. Record the input/evaluator hashes and baseline
commit. Establish a fresh baseline before calling a candidate an improvement.

Existing executable checks and entry points:

```sh
./run_hmc_solver_tests.sh
./run_hmc_solver_benchmark.sh > baseline.log 2>&1
./run_compact_hmc.sh --config /work/configs/fp4-solvers \
  --run-dir /work/runs/<unique-trial> > trajectory.log 2>&1
```

The solver benchmark is a smoke comparison; the trajectory runner already emits
wall time, stage residuals, Hamiltonians and memory. Establish a repeated baseline
and a frozen comparison receipt using these outputs as the first deliverable.
Retain the pinned container and compilation caches. Log cold startup separately.

## Experiment loop

1. Pick a testable hypothesis. Start with cheap config changes; use established
   Torch optimizers and the existing packed adapters. Research primary sources
   when they answer a concrete architecture or numerical question.
2. Publish an `idea` record, then `running` with the candidate commit and log path.
3. Make the change, run the relevant test, and execute the frozen benchmark.
   Capture exit status; inspect compact summaries and error tails, not full logs.
4. Compare repeated measurements with CG and the current best under the same
   contract. Check packed-state memory and numerical diagnostics. Run the whole
   trajectory before promoting an implementation change.
5. Publish `keep`, `discard`, or `crash`, including evidence and the reason.
   Keep a repeatable improvement; prefer simpler code when performance is tied.
   Preserve rejected commits and results without resetting shared branches.
6. Continue with the next experiment while the episode is authorized to run.
   Respect explicit stop/pause requests and runtime limits. Save a resumable
   handoff if interrupted; do not ask for approval between ordinary experiments.

Independent research and implementation may proceed concurrently in separate
worktrees. Serialize only measurements that would contend for the GPU. Delegate
bounded questions when useful, with context and an executable acceptance check.
Tanya owns the user’s 30-minute test windows and four-hour campaign deadline.
There is no GPU utilization quota. Distinguish a slow job from a stuck one.

## Evidence and continuity

Use the site's Autoresearch board as the experiment ledger. Each trial has one
stable ID; publish a new record to update it, retaining prior records. Include
contract ID, code commit, status, hypothesis, total seconds, iterations, residual,
Torch peak bytes, and links to configuration/raw results. Never invent metrics.
An outcome marked `keep` needs successful evaluation and a recorded comparison.
Use `autoresearch/git_trial.py` for isolated branches and evidence-backed keep/
discard records, or equivalent safe Git commands. See `board-record.example.json`;
post through `research_board` or the site's
record form. Keep raw artifacts and an append-only `results.tsv` with the run.

At handoff: identify the best reproducible commit, measured gains, rejected ideas,
remaining issues, next experiment, and exact resume command. Finish an episode
with results, a final review and an after-action report about the research process.
FP4 stopping guards and approximate solves do not certify exact HMC sampling.
Report reversibility, Hamiltonian changes and acceptance observations honestly.

## Source and adaptation

Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch)
and its [program.md](https://github.com/karpathy/autoresearch/blob/master/program.md),
reviewed 2026-10-29. The edit/measure/retain loop is adapted here to time-to-target,
packed storage and episodic evidence, rather than a fixed-duration LLM training run.
