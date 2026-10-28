import copy,json,tempfile
from pathlib import Path
from hmc_config import HMCConfig

cfg=HMCConfig.from_pretrained('configs/smoke')
for patch in [{'integrator':{'steps':0}},{'integrator':{'stepps':10}},{'precision':{'format':'wat'}},{'precision':{'format':'bf16','solver_components':2}},{'solvers':{'force':{'iterations':True}}},{'logging':{'tensorboard':'yes'}}]:
 try:HMCConfig.from_dict(patch)
 except ValueError:pass
 else:raise AssertionError(patch)
with tempfile.TemporaryDirectory() as td:
 cfg.save_pretrained(td);assert HMCConfig.from_pretrained(td)==cfg
print('PASS: schema validation and JSON roundtrip')

if __name__=='__main__':
 import torch
 from run_hmc_config import build_model
 from hmc_layers import HMCTrajectory
 d=cfg.to_dict();d['integrator']['steps']=2;d['diagnostics']['hamiltonian_each_stage']=True
 d['solvers']['diagnostic_energy']['iterations']=1
 small=HMCConfig.from_dict(d)
 torch.manual_seed(4);h=build_model(small);h.cold();model=HMCTrajectory(h)
 counts={'force_kick':0,'gauge_drift':0,'acceptance':0,'fermion_solver':0,'operator':0}
 modules=[(model.integrator.force_kick,'force_kick'),(model.integrator.gauge_drift,'gauge_drift'),(model.acceptance,'acceptance'),(model.fermion_solver,'fermion_solver'),(model.fermion_solver.operator,'operator')]
 for module,key in modules:
  def count(_m,_args,_out,key=key):counts[key]+=1
  module.register_forward_hook(count)
 a=model(small)
 assert counts['force_kick']==3 and counts['gauge_drift']==2 and counts['acceptance']==1,counts
 assert len(a['hamiltonian_transitions'])==5
 labels=[(r['label'],len(r['iterations'])) for r in h.solves]
 assert labels==[('initial_energy',3),('force',2),('diagnostic_energy',1),('diagnostic_energy',1),('force',2),('diagnostic_energy',1),('diagnostic_energy',1),('force',2),('final_energy',4)],labels
 assert counts['fermion_solver']==len(labels) and counts['operator']>0
 torch.manual_seed(4);base=build_model(small);base.cold()
 b=base.trajectory(2,.001,2,1,initial_energy_iterations=3,final_energy_iterations=4)
 for key in ('h_initial','h_final','delta_h','accepted'):assert a[key]==b[key],key
 torch.manual_seed(45);ten=build_model(cfg);ten.cold();HMCTrajectory(ten)(cfg)
 solves=ten.solves;assert len(solves)==13 and sum(len(z['iterations']) for z in solves)==29
 assert sum(z['label']=='force' for z in solves)==11
 assert ten.timings['drift']['calls']==10
 print('PASS: real module hooks, stage budgets, exact wrapped/unwrapped behavior, 10-step smoke counts')
