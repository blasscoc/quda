"""GPU contract tests: library equivalence, real packed storage, stopping/rollback."""
import copy
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from compact_hmc import HMC,Field
from fp4_roundoff import Expansion
from hmc_config import HMCConfig,SOLVER_DEFAULTS,merge
from hmc_solvers import PackedTorchOptimizer,StopPolicy,solve_packed,zero_vector,residual_norm
from hmc_layers import HMCTrajectory
from run_hmc_config import build_model


class DiagonalSystem(HMC):
    def __init__(self,precision='tf32',dual=False,tile=4):
        self.v=9;self.tile=tile;self.precision=precision;self.dual=dual
        self.diagnostics=False;self.residual_every=1
        self.solves=[];self.timings={};self.timing_stack=[]
        self.breakdowns=torch.zeros((),device='cuda',dtype=torch.int64)
        self.gauge=Field((1,1,4,6),precision)
        self.diagonal=torch.linspace(.5,2.,24,device='cuda').reshape(1,1,4,6)
    def vector(self):
        return Expansion((1,self.v,4,6)) if self.dual else Field((1,self.v,4,6),self.precision)
    def apply(self,x):
        y=self.vector()
        for sl,_ in self.tiles():
            ix=(slice(None),sl);y.put(ix,x.get(ix)*self.diagonal)
        return y


def run_tests():
    torch.manual_seed(72)
    # Same updates as stock Torch, including momentum initialization and each
    # tile's independent Adam step count. Last tile has a different shape.
    for algorithm in ('sgd','adam','muon'):
        h=DiagonalSystem();x=zero_vector(h);rhs=h.random_vector()
        settings=merge(SOLVER_DEFAULTS,dict(algorithm=algorithm,momentum=.8,learning_rate=.05,nesterov=True))
        adapter=PackedTorchOptimizer(h,settings)
        params=[];optimizers=[]
        for sl,_ in h.tiles():
            param=torch.nn.Parameter(torch.zeros_like(x.get((slice(None),sl))).reshape(-1,24))
            common=dict(lr=.05,weight_decay=0.)
            if algorithm=='sgd':opt=torch.optim.SGD([param],**common,momentum=.8,nesterov=True,foreach=False)
            elif algorithm=='adam':opt=torch.optim.Adam([param],**common,betas=(.9,.999),eps=1.e-8,foreach=False,fused=False)
            else:opt=torch.optim.Muon([param],**common,momentum=.8,nesterov=True,eps=1.e-8,ns_steps=5)
            params.append(param);optimizers.append(opt)
        for iteration in range(1,4):
            ax=h.apply(x);adapter.update(x,rhs,ax,iteration)
            for (sl,_),param,opt in zip(h.tiles(),params,optimizers):
                ix=(slice(None),sl)
                param.grad=(param.detach()*h.diagonal.reshape(1,24)-rhs.get(ix).reshape(-1,24))
                opt.step()
                torch.testing.assert_close(x.get(ix).reshape(-1,24),param,rtol=1.e-6,atol=1.e-7)
        assert not adapter.optimizer.state and adapter.parameter.numel()==0
    print('PASS stock Torch SGD/Adam/tile-Muon equivalence over three updates and unequal tiles')

    for algorithm in ('cg','sgd'):
        h=DiagonalSystem();rhs=h.random_vector()
        settings=dict(algorithm=algorithm,learning_rate=.5,stopping=dict(rtol=1.e-4,check_every=2,min_iterations=2))
        x=solve_packed(h,rhs,100,settings)
        record=h.solves[-1]
        assert record['stop_reason']=='converged' and record['actual_iterations']<100,record
        independent=float(torch.linalg.vector_norm((rhs.get()-h.apply(x).get()).double())/torch.linalg.vector_norm(rhs.get().double()))
        assert independent<=1.e-4 and math.isclose(independent,record['ending_relative_residual'],rel_tol=1.e-5)
        assert record['actual_iterations']%2==0
    h=DiagonalSystem();zero=zero_vector(h);solve_packed(h,zero,4)
    assert h.solves[-1]['actual_iterations']==0 and h.solves[-1]['converged']
    h=DiagonalSystem();rhs=h.random_vector()
    solve_packed(h,rhs,5,dict(algorithm='sgd',learning_rate=1.e-12,stopping=dict(stagnation_checks=2)))
    assert h.solves[-1]['stop_reason']=='stagnated' and h.solves[-1]['actual_iterations']==2
    solve_packed(h,rhs,5,dict(algorithm='sgd',learning_rate=100.,stopping=dict(divergence_factor=2.)))
    assert h.solves[-1]['stop_reason']=='diverged'
    h.diagonal.fill_(-1.)
    solve_packed(h,rhs,5)
    assert h.solves[-1]['stop_reason']=='breakdown'
    h.diagonal.fill_(float('nan'))
    solve_packed(h,rhs,5,dict(algorithm='sgd',stopping=dict(rtol=.01)))
    assert h.solves[-1]['stop_reason']=='nonfinite'
    # Diagnostics cadence must never change stop cadence or returned solution.
    a=DiagonalSystem();b=DiagonalSystem();b.diagnostics=True;b.residual_every=1
    policy=dict(algorithm='sgd',learning_rate=.5,stopping=dict(rtol=.02,check_every=4))
    rhs=a.random_vector();xa=solve_packed(a,rhs,32,policy);xb=solve_packed(b,rhs,32,policy)
    assert a.solves[-1]['actual_iterations']==b.solves[-1]['actual_iterations']
    torch.testing.assert_close(xa.q,xb.q,rtol=0,atol=0)
    print('PASS convergence, minimum/check cadence, zero RHS, stagnation, divergence, breakdown, nonfinite')

    for precision,dual in [('fp4',False),('fp4',True),('fp8',False),('fp16',False),('bf16',False),('tf32',False)]:
        for algorithm in ('cg','sgd','adam','muon'):
            h=DiagonalSystem(precision,dual);rhs=h.random_vector()
            x=solve_packed(h,rhs,3,dict(algorithm=algorithm,momentum=.9,learning_rate=.05,stopping=dict(rtol=1.e-8)))
            assert math.isfinite(h.solves[-1]['ending_relative_residual'])
            assert x.q.dtype==rhs.q.dtype and x.bytes()==rhs.bytes()
            if precision=='fp4':
                assert x.q.dtype==torch.uint8 and x.bytes()==9*4*(8 if dual else 4)
                expected_fields=3 if algorithm in ('cg','adam') else 2
                assert h.solves[-1]['persistent_state_bytes']==expected_fields*x.bytes()
    print('PASS all solvers/all formats and both FP4 components; packed state byte accounting')

    base=HMCConfig.from_pretrained('configs/smoke').to_dict()
    base['integrator']['steps']=2
    base['diagnostics'].update(enabled=False,hamiltonian_each_stage=False)
    base['solvers']['force'].update(algorithm='sgd',learning_rate=.1)
    base['solvers']['final_energy'].update(algorithm='adam',learning_rate=.1)
    base['solvers']['initial_energy'].update(algorithm='muon',learning_rate=.05,momentum=.9)
    cfg=HMCConfig.from_dict(base);h=build_model(cfg);h.cold();model=HMCTrajectory(h)
    result=model(cfg)
    assert result['completed_steps']==2 and not result['aborted']
    assert [r['algorithm'] for r in h.solves]==['muon','sgd','sgd','sgd','adam']
    # Hard failure during the second force occurs AFTER a gauge drift. Rollback
    # must restore both packed gauge bits and the operator cache.
    original=h.gauge.clone();probe=h.random_vector();before=h.apply(probe).get().clone()
    raw=model.integrator.force_kick.forward;calls=0
    def fail(*args):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('injected kernel failure')
        return raw(*args)
    model.integrator.force_kick.forward=fail
    try:model(cfg)
    except RuntimeError as exc:assert str(exc)=='injected kernel failure'
    else:raise AssertionError('Expected failure')
    assert torch.equal(h.gauge.q,original.q) and torch.equal(h.gauge.s,original.s)
    torch.testing.assert_close(h.apply(probe).get(),before,rtol=0,atol=0)
    # Actual packed centered-gauge trajectory + active energy abort with logs off.
    guards=copy.deepcopy(base);guards['precision'].update(solver_components=2,gauge_representation='identity_plus_correction',momentum_components=2)
    for s in guards['solvers'].values():s['algorithm']='cg'
    guards['integrator']['stopping']['max_abs_delta_h']=1.e-12
    cfg=HMCConfig.from_dict(guards);h=build_model(cfg);h.cold();old=h.gauge.clone()
    result=HMCTrajectory(h)(cfg)
    assert result['aborted'] and result['stop_reason']=='energy_guard' and not result['accepted'],result
    assert result['completed_steps']==1 and result['h_final'] is None and result['integrator_guard_checks']
    assert torch.equal(h.gauge.q,old.q) and torch.equal(h.gauge.s,old.s)
    # Unconverged initial energy is rejected before integration, even logging off.
    guards['integrator']['stopping'].update(max_abs_delta_h=None,reject_on_unconverged=True)
    for s in guards['solvers'].values():s['stopping']['rtol']=1.e-12
    cfg=HMCConfig.from_dict(guards);h=build_model(cfg);h.cold()
    result=HMCTrajectory(h)(cfg)
    assert result['aborted'] and result['stop_reason']=='solver_unconverged' and result['completed_steps']==0
    print('PASS stage dispatch, actual HMC execution, exception rollback, energy abort, unconverged rejection')

    for patch in [dict(solvers=dict(force=dict(algorithm='foo'))),dict(solvers=dict(force=dict(stopping=dict(rtol=-1)))),
                  dict(solvers=dict(force=dict(algorithm='adam',residual_refresh_every=1))),
                  dict(integrator=dict(stopping=dict(reject_on_unconverged=True)))]:
        try:HMCConfig.from_dict(patch)
        except ValueError:pass
        else:raise AssertionError(patch)
    print('PASS invalid policy/config rejection')

    # Exercise the public CLI, actual checkpoint loader and emitted event files.
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        def run(name,count,resume=None):
            args=[sys.executable,'run_hmc_config.py','--config','configs/fp4-solvers',
                  '--set','physics.lattice=[4,4,4,4]','--set',f'execution.trajectories={count}',
                  '--set','logging.tensorboard=true','--run-dir',str(root/name)]
            if resume:args+=['--resume',str(root/resume/'checkpoint.pt')]
            proc=subprocess.run(args,text=True,capture_output=True)
            assert proc.returncode==0,proc.stdout[-3000:]+proc.stderr
        run('continuous',2);run('first',1);run('resumed',1,'first')
        a=torch.load(root/'continuous/checkpoint.pt',map_location='cpu',weights_only=True)
        b=torch.load(root/'resumed/checkpoint.pt',map_location='cpu',weights_only=True)
        for key in ('gauge_q','gauge_s','cpu_rng','cuda_rng'):assert torch.equal(a[key],b[key]),key
        assert a['completed']==b['completed']==2
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events=EventAccumulator(str(root/'resumed/tensorboard')).Reload()
        tags=events.Tags()
        assert 'solve_summary/force/actual_iterations' in tags['scalars']
        force_events=events.Scalars('solve_summary/force/actual_iterations')
        assert len(force_events)==2 and force_events[0].step<force_events[1].step
        assert 'trajectory/stop_reason/text_summary' in tags['tensors']
        result=json.loads((root/'resumed/results.jsonl').read_text().splitlines()[-1])
        assert result['solves'][1]['algorithm']=='sgd'
    print('PASS mixed-solver CLI checkpoint/resume equivalence and TensorBoard summaries')

if __name__=='__main__':run_tests()
