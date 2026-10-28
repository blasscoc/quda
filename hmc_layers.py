"""Inspectable PyTorch module graph around existing HMC implementations.

These are stateful physics layers, not learned layers. Only the smooth force
surrogate uses autograd. Packing, CG control and Metropolis are not end-to-end
 differentiable. State is serialized by the explicit checkpoint API, not state_dict.
"""
import math
import json
import time
import torch
from torch import nn

class BackendLayer(nn.Module):
    def __init__(self,backend):
        super().__init__()
        # Backend owns packed simulation state; do not register duplicate buffers.
        object.__setattr__(self,'backend',backend)
    def extra_repr(self):return 'packed state owned by backend'

class FermionOperator(BackendLayer):
    def forward(self,x):return self.backend.apply(x)

class FermionSolve(BackendLayer):
    def __init__(self,backend):
        super().__init__(backend);self.operator=FermionOperator(backend)
    def extra_repr(self):
        policies=getattr(self.backend,'solver_policies',{})
        return ', '.join(f"{stage}={s['algorithm']}({s['iterations']} max)" for stage,s in policies.items()) or 'legacy CG'
    def forward(self,rhs,iterations):
        result=self.backend.solve(rhs,iterations)
        record=self.backend.solves[-1]
        guards=getattr(self.backend,'integrator_stopping',{})
        # Diagnostic solves observe the proposal, they do not define it.
        if record['label']!='diagnostic_energy':
            failure=record['stop_reason'] in ('nonfinite','breakdown','diverged','stagnated')
            if guards.get('reject_on_solver_failure') and failure:
                raise TrajectoryAbort('solver_'+record['stop_reason'])
            if guards.get('reject_on_unconverged') and record['converged'] is not True:
                raise TrajectoryAbort('solver_unconverged')
        return result

class MomentumRefresh(BackendLayer):
    def forward(self):return self.backend.momentum()

class PseudofermionRefresh(BackendLayer):
    def forward(self):return self.backend.schur(self.backend.random_vector(),dagger=True)

class ForceKick(BackendLayer):
    def forward(self,mom,phi,dt,iterations):return self.backend.kick(mom,phi,dt,iterations)

class GaugeDrift(BackendLayer):
    def forward(self,mom,dt):return self.backend.drift(mom,dt)

class Hamiltonian(BackendLayer):
    def forward(self,phi,mom,iterations):return self.backend.energy(phi,mom,iterations)

class LeapfrogIntegrator(BackendLayer):
    def __init__(self,backend):
        super().__init__(backend);self.force_kick=ForceKick(backend);self.gauge_drift=GaugeDrift(backend)
    def forward(self,mom,phi,steps,dt,iterations,observe):
        for i in range(steps):
            observe(f'kick_{i+1}',lambda:self.force_kick(mom,phi,dt*(.5 if i==0 else 1.),iterations))
            observe(f'drift_{i+1}',lambda:self.gauge_drift(mom,dt))
        observe('final_half_kick',lambda:self.force_kick(mom,phi,dt*.5,iterations))

class Metropolis(nn.Module):
    def forward(self,h0,h1):
        delta=float(h1-h0)
        accepted=math.isfinite(delta) and math.log(max(float(torch.rand((),device=h0.device)),1.e-30)) < -delta
        return delta,accepted

class TrajectoryAbort(RuntimeError):
    """Configured experiment guard: roll back the whole proposal, never accept a prefix."""

class HMCTrajectory(BackendLayer):
    def __init__(self,backend):
        super().__init__(backend)
        self.momentum_refresh=MomentumRefresh(backend)
        self.pseudofermion_refresh=PseudofermionRefresh(backend)
        self.fermion_solver=FermionSolve(backend)
        backend.solver_layer=self.fermion_solver
        backend.operator_layer=self.fermion_solver.operator
        self.integrator=LeapfrogIntegrator(backend)
        self.hamiltonian=Hamiltonian(backend)
        self.acceptance=Metropolis()
        self.config=None
    @classmethod
    def from_config(cls,config,initialize=True):
        from run_hmc_config import build_model
        torch.manual_seed(config.data['seed'])
        torch.backends.cuda.matmul.allow_tf32=config.data['precision']['format']=='tf32'
        model=cls(build_model(config));model.config=config
        if initialize:model.backend.cold()
        return model
    def forward(self,config=None):
        config=config or self.config
        if config is None:raise ValueError('Supply HMCConfig or construct with HMCTrajectory.from_config')
        d=config.data;s=d['solvers'];i=d['integrator']
        built=getattr(self.backend,'configuration',None)
        if built and (d['physics']!=built['physics'] or d['precision']!=built['precision'] or
                      d['execution']['tile_sites']!=built['execution']['tile_sites']):
            raise ValueError('Physics/precision/tile changes require constructing a new workflow')
        self.backend.solver_policies=s
        self.backend.integrator_stopping=i['stopping']
        return self.backend.trajectory(i['steps'],i['step_size'],s['force']['iterations'],s['diagnostic_energy']['iterations'],
            initial_energy_iterations=s['initial_energy']['iterations'],final_energy_iterations=s['final_energy']['iterations'],
            diagnostic_energy_iterations=s['diagnostic_energy']['iterations'],layers=self)

    def run(self,steps,dt,force_iterations,initial_iterations,final_iterations,diagnostic_iterations):
        """Own the refresh -> integrate -> measure -> accept/rollback transaction.

Guards are optional research controls, not an exact adaptive HMC construction.
The leapfrog module owns the schedule; physics kernels own packed state only.
"""
        h=self.backend;backup=h.gauge.clone();transitions=[];guard_checks=[]
        guards=getattr(h,'integrator_stopping',{})
        limit=guards.get('max_abs_delta_h')
        cadence=guards.get('check_every_steps',1)
        trace=h.diagnostics and getattr(h,'hamiltonian_each_stage',True)
        completed=0;stage='refresh';h0=h1=last_h=None;previous=None
        delta=None;accepted=False;stop_reason='completed';aborted=False

        def advance(label,operation):
            nonlocal previous,last_h,completed,stage
            stage=label
            torch.cuda.synchronize();start=time.perf_counter()
            operation()
            torch.cuda.synchronize();operation_seconds=time.perf_counter()-start
            if label.startswith('drift_'):completed+=1
            check_guard=limit is not None and (label=='final_half_kick' or
                (label.startswith('drift_') and completed%cadence==0))
            if trace or check_guard:
                start=time.perf_counter()
                h.energy_stage='final_energy' if label=='final_half_kick' else 'diagnostic_energy'
                last_h=self.hamiltonian(phi,mom,final_iterations if label=='final_half_kick' else diagnostic_iterations)
                torch.cuda.synchronize();diagnostic_seconds=time.perf_counter()-start
                after=dict(h.energy_components) if h.diagnostics else {'total':float(last_h)}
                if trace:
                    row=dict(stage=label,before=previous,after=after,delta_h=after['total']-previous['total'],
                             operation_seconds=operation_seconds,hamiltonian_diagnostic_seconds=diagnostic_seconds)
                    transitions.append(row)
                    print(json.dumps(dict(event='hamiltonian_transition',**row)),flush=True)
                previous=after
                if check_guard:
                    change=float(last_h-h0)
                    guard_checks.append(dict(stage=label,delta_h=change,seconds=diagnostic_seconds))
                    if not math.isfinite(change):raise TrajectoryAbort('nonfinite_energy')
                    if abs(change)>limit:raise TrajectoryAbort('energy_guard')

        try:
            mom=self.momentum_refresh();phi=self.pseudofermion_refresh()
            stage='initial_energy';h.energy_stage=stage
            h0=self.hamiltonian(phi,mom,initial_iterations)
            if not math.isfinite(float(h0)):raise TrajectoryAbort('nonfinite_energy')
            previous=dict(h.energy_components) if h.diagnostics else {'total':float(h0)}
            last_h=h0
            self.integrator(mom,phi,steps,dt,force_iterations,advance)
            stage='final_energy';h.energy_stage=stage
            # The final half kick is measured by both trace and guard paths.
            h1=last_h if trace or limit is not None else self.hamiltonian(phi,mom,final_iterations)
            last_h=h1
            if not math.isfinite(float(h1)):raise TrajectoryAbort('nonfinite_energy')
            delta,accepted=self.acceptance(h0,h1)
        except TrajectoryAbort as exc:
            aborted=True;stop_reason=str(exc)
        except BaseException:
            # Errors must not leave a partially integrated proposal as chain state.
            h.gauge=backup;h.refresh()
            raise
        if not accepted:
            h.gauge=backup;h.refresh()
        return dict(h_initial=float(h0) if h0 is not None else None,
                    h_final=float(h1) if h1 is not None else None,delta_h=delta,accepted=accepted,
                    hamiltonian_transitions=transitions,completed_steps=completed,requested_steps=steps,
                    aborted=aborted,stop_reason=stop_reason,stop_stage=stage,
                    integrator_guard_checks=guard_checks,
                    last_observed_hamiltonian=float(last_h) if last_h is not None else None)
