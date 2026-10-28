"""Render saved solver measurements without rerunning the GPU."""
import csv,json
from pathlib import Path
r=json.loads(Path('solver_tolerances.json').read_text())
if not r.get('completed'):raise SystemExit('Benchmark is still running; read solver_tolerances.json for live progress')
s=r['sgd'];rows=[]
for c in r['cg']:
    rows.append(['CG FP64',c['target'],c['reached'],c['iterations'],c['mean_internal_s_per_iteration'],c['solve_s'],c['validated_wall_s'],c['true_residual']])
for target in r['targets']:
    hit=s['targets'][str(target)]
    rows.append(['FP4 SGD',target,hit is not None,hit['step'] if hit else '',s['mean_step_s'],
                 hit['update_total_s'] if hit else '',hit['validated_wall_s'] if hit else '',
                 hit['true_residual'] if hit else s['final_true_residual']])
with Path('solver_tolerances.csv').open('w') as f:
    w=csv.writer(f,lineterminator='\n')
    w.writerow(['solver','target','reached','iterations_to_target','mean_step_s',
                'solve_or_update_s_to_target','validated_wall_s_to_target','true_residual'])
    w.writerows(rows)
best=min(s['history'],key=lambda h:h['true_residual'])
lines=['CG FP64 versus compiled FP4 approximate-gradient SGD',
       '56^3 x 96, same gauge and seed-45 RHS, batch 1, zero initial guess.',
       'True residual: ||b-A_FP64*x||/||b||. Setup and compilation excluded.',
       '', 'CG target | iterations | mean internal seconds/iteration | solve seconds | validated seconds']
for c in r['cg']:
    lines.append(f"{c['target']:.0e} | {c['iterations']} | {c['mean_internal_s_per_iteration']:.4f} | {c['solve_s']:.4f} | {c['validated_wall_s']:.4f}")
lines+=['',f"FP4 SGD: {s['iterations']} steps, mean {s['mean_step_s']:.4f} s/update, median {s['median_step_s']:.4f} s/update.",
        f"Update-only total {s['update_total_s']:.4f} s; checks {s['check_total_s']:.4f} s; validated elapsed {s['validated_wall_s']:.4f} s.",
        f"Best true residual {best['true_residual']:.8g} at step {best['step']}; final {s['final_true_residual']:.8g}.",
        f"Stop: {s['stop_reason']}",f"Learning rate {s['lr']:.8g}; vanilla torch.optim.SGD, no momentum or weight decay.",
        'FP4 products use FP32 accumulation/state. The approximate A_fp4(x)-b direction',
        'is supplied explicitly; no claim of differentiating the quantized network.',
        'CG step time is internal solver time divided by iteration count; SGD steps',
        'are individually synchronized wall times. Both include their vector updates.',
        'SGD true residual checks run every step and are timed separately. CG is checked',
        'after each target solve. Validated wall times therefore include different',
        'monitoring overhead; use the update/solve column to compare algorithm cost.',
        'This is one SGD learning-rate/configuration trial, not proof that all FP4',
        'solvers or learned preconditioners cannot converge.',
        'Raw histories, target crossings, setup and calibration times: solver_tolerances.json.']
Path('SOLVER_RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
