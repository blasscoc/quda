import json,torch
from fp4_roundoff import ResearchHMC,Expansion,StochasticField,reference_apply
from wilson_network import WilsonNormalNetwork
from demo_fp4_roundoff import seed_gauge

torch.manual_seed(4);torch.backends.cuda.matmul.allow_tf32=False
rows=[]
for mode in ('baseline','dual','center','center_dual'):
 for warm in (False,True):
  h=ResearchHMC([4]*4,mode,tile=64);seed_gauge(h,warm)
  raw=torch.randn(1,h.v,4,6,device='cuda');x=h.vector();x.put(...,raw)
  u=torch.stack([torch.stack([torch.view_as_real(h.u(mu,p,torch.arange(h.v,device='cuda'))) for p in (0,1)]) for mu in range(4)])
  net=WilsonNormalNetwork(u,h.dims,h.kappa)
  exact=net(x.get().reshape(1,h.v,4,3,2)).flatten(-2)
  y=h.apply(x).get();ref=reference_apply(h,x).get()
  error=float((y-exact).norm()/exact.norm());oracle=float((ref-exact).norm()/exact.norm())
  assert error < (.025 if h.dual else .4),(mode,warm,error)
  assert oracle<.002,oracle
  rows.append(dict(mode=mode,warm=warm,operator_error=error,diagnostic_reference_error=oracle))
# Test scale preservation at tiny magnitude; no volume-sized floating master.
x=torch.randn(1,128,4,6,device='cuda')*1.e-5
plain=Expansion(x.shape);scaled=Expansion(x.shape,65536.)
plain.put(...,x);scaled.put(...,x)
a=float((plain.get()-x).norm()/x.norm());b=float((scaled.get()-x).norm()/x.norm())
assert b<a/5 and b<.02,(a,b)
# SR expectation check on values safely inside representable scale range.
x=torch.full((65536,6),1.1,device='cuda');x[:,0]=6.
f=StochasticField(x.shape,'fp4');f.put(...,x)
bias=float((f.get()[:,1:]-x[:,1:]).mean());assert abs(bias)<.003,bias
print(json.dumps(dict(operator_checks=rows,small_rhs_errors=[a,b],stochastic_bias=bias)),flush=True)
from fp4_momentum import CompactMomentum
import tempfile
from pathlib import Path
p=CompactMomentum((4,2,128,2,6))
a=torch.randn(4,2,128,12,device='cuda');a[...,8:]=0
p.put(...,a.reshape(4,2,128,2,6))
assert p.bytes()==4*2*128*10
assert float((p.get().flatten(-2)-a).norm()/a.norm())<.025
h=ResearchHMC([4]*4,'center_dual_mom',tile=64)
h.cold();h.diagnostics=True;h.stable_energy=True;h.residual_refresh_interval=2
with tempfile.TemporaryDirectory() as td:
 path=Path(td)/'state.pt';h.checkpoint(path,{},0)
 result=h.trajectory(1,.001,2,2)
 expected=h.gauge.q.clone();es=h.gauge.s.clone()
 restored=ResearchHMC.restore(path);restored.diagnostics=True
 repeated=restored.trajectory(1,.001,2,2)
 for key in ('h_initial','h_final','delta_h','accepted'):assert result[key]==repeated[key],key
 assert torch.equal(expected,restored.gauge.q)
 assert torch.equal(es.float(),restored.gauge.s.float())
 assert len(result['hamiltonian_transitions'])==3
print('PASS: operator/reference, scaling, SR bias, compact momentum, stage energies, checkpoint/restart')
