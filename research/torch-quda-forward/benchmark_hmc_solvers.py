"""Short 32^4 packed-solver comparison; no QUDA or full precision lattice copy."""
import json
import gc
from pathlib import Path
import torch
from hmc_config import HMCConfig,revision
from hmc_layers import HMCTrajectory
from hmc_solvers import solve_packed
from run_hmc_config import clean

def main():
    rows=[]
    for components in (1,2):
        cfg=HMCConfig.from_dict(dict(physics=dict(lattice=[32]*4),
            precision=dict(solver_components=components,gauge_representation='identity_plus_correction'),
            diagnostics=dict(enabled=True,residual_every=4,hamiltonian_each_stage=False)))
        torch.manual_seed(45);model=HMCTrajectory.from_config(cfg);h=model.backend;rhs=h.random_vector()
        for algorithm in ('cg','sgd','adam','muon'):
            settings=dict(algorithm=algorithm,learning_rate=.25 if algorithm=='sgd' else .1,
                          momentum=.9 if algorithm=='muon' else 0.,
                          epsilon=.01 if algorithm=='adam' else 1.e-8,
                          stopping=dict(rtol=.1,check_every=2,stagnation_checks=4,divergence_factor=10.))
            # Compile/warm exactly this path outside the timed measured solve.
            h.diagnostics=False;warm=solve_packed(h,rhs,1,settings);del warm
            h.diagnostics=True;h.solves=[];h.timings={};torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            x=solve_packed(h,rhs,24,settings)
            record=h.solves[-1]
            record.update(components=components,precision='fp4',lattice=[32]*4,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
            rows.append(record);del x
        del rhs,model,h
        gc.collect()
        torch.cuda.empty_cache()
    cfg=HMCConfig.from_pretrained('configs/fp4-solvers')
    torch.manual_seed(45);model=HMCTrajectory.from_config(cfg);h=model.backend
    torch.cuda.reset_peak_memory_stats();result=model()
    result.update(solves=h.solves,timings=h.timings,peak_allocated_bytes=torch.cuda.max_memory_allocated(),config=cfg.to_dict())
    output=dict(code=revision(),torch_version=str(torch.__version__),solvers=rows,trajectory=result,
                scope='Warm 32^4 cold-gauge smoke, untuned learning rates; packed-operator residual, not accuracy certification')
    path=Path('benchmark-results/hmc-solvers-32.json');path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(clean(output),indent=2,allow_nan=False)+'\n')
    for r in rows:
        print(json.dumps({k:r[k] for k in ('algorithm','components','actual_iterations','seconds_per_iteration','ending_relative_residual','stop_reason','persistent_state_bytes','peak_allocated_bytes')}),flush=True)
    print('32^4 trajectory:',result['stop_reason'],result['completed_steps'],result['accepted'],flush=True)

if __name__=='__main__':main()
