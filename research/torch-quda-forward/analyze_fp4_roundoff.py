"""Create reproducible tables/report/figure from completed GPU experiment JSON."""
import csv,json,statistics
from pathlib import Path
root=Path('benchmark-results/fp4-roundoff')
records={p.stem:json.loads(p.read_text()) for p in root.glob('*.json') if p.stem not in ('smoke','verification')}
rows=[]
for name,r in sorted(records.items()):
 if r['kind']!='solver':continue
 rows.append(dict(case=name,iterations=r['iterations'],initial_loss=1.,final_native_loss=r['native_residual']**2,
  independent_residual=r['independent_residual'],independent_loss=r['independent_residual']**2,
  rhs_storage_error=r['rhs_storage_error'],seconds_per_iteration=r['mean_iteration_seconds'],
  wall_seconds=r['wall_seconds'],diagnostic_seconds=r['logged_diagnostic_seconds'],peak_mib=r['peak_allocated_bytes']/2**20))
with open('FP4_ROUNDOFF_SOLVERS.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
lines=['FP4 roundoff results: 32^4 GPU experiments','=========================================',
 'All solver residuals below use the independent FP32-tile/FP16-storage diagnostic.',
 'That reference agreed with the full FP32 network to about 3.2e-4 on tiny cold/warm tests.',
 'References use each mode\'s decoded gauge and RHS; RHS quantization errors are separately recorded.',
 'All modes start from the same raw RHS. SR random draws use a separate RNG from RHS generation.',
 'Initial normalized residual-squared loss is 1. See CSV for every run.','',
 'case                         iters   end residual    mean s/iter   solve+diagnostics s   peak MiB']
for row in rows:lines.append(f"{row['case']:28s} {row['iterations']:5d} {row['independent_residual']:14.6f} {row['seconds_per_iteration']:14.6f} {row['wall_seconds']:21.4f} {row['peak_mib']:10.2f}")
lines+=['','Trajectory comparisons: one leapfrog step, dt=.001, four iterations per solve.',
 'mode/seed                       DeltaH         accepted   peak MiB   changed links/total']
for name in ('trajectory-baseline','baseline-seed-46','baseline-seed-47','trajectory-center_dual','trajectory-center_dual_mom','trajectory-stable','compact-momentum-45','compact-momentum-46','compact-momentum-47'):
 r=records[name];m=r['motion'][0];z=r['result'];lines.append(f"{name:30s} {z['delta_h']:12.5f} {str(z['accepted']):>8s} {r['peak_allocated_bytes']/2**20:10.2f} {m['changed_links']}/{m['total_links']}")
lines+=['','Per-stage Hamiltonian, compact eight-coordinate momentum, seed 45:']
r=records['compact-momentum-45']
for t in r['result']['hamiltonian_transitions']:
 lines.append(f"{t['stage']}: H {t['before']['total']:.8f} -> {t['after']['total']:.8f}; delta {t['delta_h']:.8f}; operation {t['operation_seconds']:.5f}s; energy diagnostic {t['hamiltonian_diagnostic_seconds']:.5f}s")
lines+=['','Conclusions:',
 '* Two-component FP4 gives about an order-of-magnitude lower operator/storage',
 '  error, at roughly twice the solve iteration cost in this prototype.',
 '* Recomputing/restarting residual every four iterations improves the 24-step',
 '  result over the matched 24-step no-refresh control, but a ~2.4% residual',
 '  floor remains. No 1e-2 or 1e-3 convergence claim is justified.',
 '* Tiny-RHS scaling is essential: gain 65536 at amplitude 1e-5 restores useful',
 '  behavior. Gain is supplied explicitly, not automatically tuned.',
 '* State-only stochastic rounding worsened this solver in the tested cases.',
 '* The direct gauge representation froze every link at this dt. Centered gauge',
 '  storage plus a matching native cache/kernel preserved all link updates.',
 '* Momentum compensation greatly reduced |DeltaH|, but the remaining drift',
 '  varied widely across seeds. One accepted proposal in three is NOT an',
 '  acceptance-rate estimate or evidence of detailed balance/physical validity.',
 '* Scalar FP64 reduction changed the padded-momentum seed-45 DeltaH from 60',
 '  to about 64.54. Summation error matters, but does not explain all the drift.',
 '* The compact-momentum variant changes block boundaries (8 coefficients per',
 '  scale instead of two padded groups of 6). Its energy differs from the padded',
 '  variant; this is an accuracy/memory tradeoff, not a bit-identical packing.',
 '',
 '128^4 memory model (not run):',
 '  Existing FP4 HMC volume-dependent peak: approximately 80 GiB.',
 '  Second components for seven live solver vectors: +14 GiB.',
 '  Compact two-component momentum: 10 GiB vs previous 8 GiB, hence +2 GiB.',
 '  Projected tensor peak: about 96 GiB, plus bounded tile/library workspace.',
 '  Measured candidate peak at 32^4: 448.009 MiB. Subtracting the observed',
 '  ~64 MiB fixed workspace before volume scaling agrees with this model.',
 '  Blindly scaling the whole 32^4 peak yields 112 GiB and overcounts that fixed',
 '  workspace. Neither estimate is a substitute for a larger measurement.',
 '  Checkpoint staging may add ~12 GiB, leaving only modest OS/driver headroom',
 '  in 119.68 GiB shared RAM. 128^4 remains a target, not a demonstrated capacity',
 '  for this new recipe. Streamed checkpoint serialization is a sensible next step.',
 '',
 'Limitations and next experiments:',
 '  No thermalized configurations, physical-mass claim, exact reversibility, or',
 '  sampling correctness established. Further investigate gauge coordinates that',
 '  enforce SU(3), adaptive per-vector scale metadata, and a smoother/more accurate',
 '  operator before interpreting Hamiltonian rejection as only a step-size issue.',
 '  Do not absorb quantization bias into a rescaled acceptance exponent.',
 '  No MS-EDEN/Hadamard implementation or learned preconditioner is claimed.',
 '  Native tensor-core products retain FP32 accumulation; energy-only scalar',
 '  reductions optionally use FP64. There is no lattice-sized FP32 master state.',
 '',
 'Tests: run_fp4_roundoff_tests.sh and run_compact_hmc_tests.sh passed.',
 'Formalism, workflow mapping, bibliography and commands: FP4_ROUNDOFF_RESEARCH.txt.']
Path('FP4_ROUNDOFF_RESULTS.txt').write_text('\n'.join(lines)+'\n')
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,5))
 for name,label in [('cold-baseline','Single FP4'),('cold-sr','State stochastic rounding'),('cold-center_dual','Two-component FP4'),('cold-refresh','Two components + restart every 4')]:
  r=records[name];h=r['history'];ax.semilogy([0]+[z['iteration'] for z in h],[1]+[z['recomputed_relative_residual']**2 for z in h],label=label,marker='.',linewidth=1.5)
 ax.set(xlabel='Fermion solver iteration',ylabel='Recomputed native residual squared / RHS norm squared',title='32⁴ FP4 roundoff ablations (cold gauge)')
 ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig('FP4_ROUNDOFF_CONVERGENCE.svg');plt.close(fig)
except ImportError:
 print('Matplotlib unavailable; tables/report generated.')
