"""Inspectable PyTorch module graph around existing HMC implementations.

These are stateful physics layers, not learned layers. Only the smooth force
surrogate uses autograd. Packing, CG control and Metropolis are not end-to-end
 differentiable. State is serialized by the explicit checkpoint API, not state_dict.
"""
import math
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
    def forward(self,rhs,iterations):return self.backend.solve(rhs,iterations)

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
    def forward(self,config):
        d=config.data;s=d['solvers'];i=d['integrator']
        return self.backend.trajectory(i['steps'],i['step_size'],s['force']['iterations'],s['diagnostic_energy']['iterations'],
            initial_energy_iterations=s['initial_energy']['iterations'],final_energy_iterations=s['final_energy']['iterations'],
            diagnostic_energy_iterations=s['diagnostic_energy']['iterations'],layers=self)
