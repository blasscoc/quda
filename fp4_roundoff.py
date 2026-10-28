"""Experimental FP4 roundoff treatments. Volume-sized data remain packed.
Two components are explicitly ~10.67 effective bits/real, NOT free FP4 accuracy.
"""
import math
import torch
import triton
from compact_hmc import HMC, Field, timed
from packed_fp4 import pack_rows, unpack_rows
from wilson_network import complex_matrix_as_real, WilsonNormalNetwork
from fp4_research_kernel import _research_dslash
from fp4_momentum import CompactMomentum

class Expansion(Field):
    def __init__(self, shape, gain=1.):
        super().__init__(shape, 'fp4')
        self.low = Field(shape, 'fp4')
        self.gain = gain
        self.low.q.zero_(); self.low.s.fill_(1)
    def get(self, index=...):
        return (super().get(index)+self.low.get(index)/256.)/self.gain
    def put(self, index, value):
        scaled=value*self.gain
        super().put(index, scaled)
        self.low.put(index, (scaled-super().get(index))*256.)
    def clone(self):
        out=Expansion(self.shape,self.gain)
        out.q.copy_(self.q);out.s.copy_(self.s)
        out.low.q.copy_(self.low.q);out.low.s.copy_(self.low.s)
        return out
    def bytes(self):return super().bytes()+self.low.bytes()

class StochasticField(Field):
    def put(self,index,value):
        scale=(value.abs().amax(-1)/6).clamp_min(2**-9).log2().ceil().exp2()
        z=value/scale[...,None]
        levels=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=value.device)
        high=torch.bucketize(z.abs().contiguous(),levels).clamp(1,7)
        lo,hi=levels[high-1],levels[high]
        prob=(z.abs()-lo)/(hi-lo)
        rounded=torch.where(torch.rand_like(z)<prob,hi,lo)*z.sign()
        # Repacking exact representable levels preserves chosen stochastic value.
        q,s=pack_rows(rounded*scale[...,None])
        self.q[index]=q;self.s[index]=s
    def clone(self):
        out=StochasticField(self.shape,'fp4');out.q.copy_(self.q);out.s.copy_(self.s);return out

class CenteredGauge(Field):
    """U=I+Q(128*(U-I))/128, same bytes as direct FP4 gauge."""
    def __init__(self, shape):
        super().__init__(shape,'fp4')
        self.eye=torch.view_as_real(torch.eye(3,device='cuda',dtype=torch.complex64)).reshape(3,6)
    def get(self,index=...):return self.eye+super().get(index)/128.
    def put(self,index,value):super().put(index,(value-self.eye)*128.)
    def clone(self):
        out=CenteredGauge(self.shape);out.q.copy_(self.q);out.s.copy_(self.s);return out

class ResearchHMC(HMC):
    def __init__(self,dims,mode='dual',gain=1.,**kwargs):
        super().__init__(dims,'fp4',**kwargs)
        self.mode,self.gain=mode,gain
        self.dual=mode in ('dual','center_dual','center_dual_mom')
        self.center=mode in ('center','center_dual','center_dual_mom')
        if self.center:self.gauge=CenteredGauge(self.gauge.shape)
    def vector(self):
        if self.dual:return Expansion((1,self.v,4,6),self.gain)
        if self.mode=='sr':return StochasticField((1,self.v,4,6),'fp4')
        return super().vector()
    @timed
    def refresh(self):
        if not self.center:return super().refresh()
        eye=torch.eye(3,device='cuda',dtype=torch.complex64)
        for p in (0,1):
            for sl,ids in self.tiles():
                for mu in range(4):
                    for back in (0,1):
                        j=self.op.indices[p,mu,1,sl].long() if back else ids
                        u=self.u(mu,1-p if back else p,j)
                        if back:u=u.mH
                        delta=(u-eye)*128.
                        w=complex_matrix_as_real(torch.view_as_real(delta.resolve_conj()))[:,::2,:]
                        self.weights.put((p,2*mu+back,sl),w)
    def dslash(self,x,parity,dagger=False,skip=None):
        if not self.dual and not self.center:return super().dslash(x,parity,dagger,skip)
        y=self.vector(); sk=x if skip is None else skip
        xl=x.low if self.dual else x;yl=y.low if self.dual else y;sl=sk.low if self.dual else sk
        _research_dslash[(triton.cdiv(self.v,4),)](
            self.op.weights[parity],self.op.scales[parity],x.q,x.s,xl.q,xl.s,
            self.op.indices[parity],self.op.signs[parity],sk.q,sk.s,sl.q,sl.s,
            y.q,y.s,yl.q,yl.s,self.v,1,16,dagger,skip is not None,self.kappa**2,
            4,8,self.dual,self.center,num_warps=1,enable_fp_fusion=False)
        return y
    @timed
    def apply(self,x):return self.schur(self.schur(x),True)
    # Parent solve allocates x via vector() then zeros q/s only. Initialize low too.
    # Expansion() initializes its low component below; outputs overwrite it.
    @timed
    def momentum(self):
        if self.mode!='center_dual_mom':return super().momentum()
        p=(Expansion if getattr(self,'padded_momentum',False) else CompactMomentum)((4,2,self.v,2,6))
        for sl,ids in self.tiles():
            for mu in range(4):
                for parity in (0,1):
                    a=torch.zeros(len(ids),12,device='cuda')
                    a[:,:8]=torch.randn(len(ids),8,device='cuda')
                    p.put((mu,parity,sl),a.reshape(-1,2,6))
        return p

    @timed
    def energy(self,phi,mom,iterations):
        if not getattr(self,'stable_energy',False):return super().energy(phi,mom,iterations)
        self.solve_label='hamiltonian'
        x=self.solve(phi,iterations)
        self.energy_reduction_dtype=torch.float64
        try:
            g,k,f=self.gauge_action(),self.kinetic(mom),self.dot(phi,x)
        finally:
            self.energy_reduction_dtype=torch.float32
        total=g+k+f
        self.energy_components=dict(gauge=float(g),kinetic=float(k),fermion=float(f),total=float(total))
        return total

    def checkpoint(self,path,config,completed):
        # Solver workspaces are transient; retain representation metadata and RNG.
        torch.save(dict(format='research-fp4-v1',mode=self.mode,gain=self.gain,dims=self.dims,
            kappa=self.kappa,beta=self.beta,tile=self.tile,boundary=self.boundary,
            gauge_q=self.gauge.q,gauge_s=self.gauge.s,rng=torch.cuda.get_rng_state(),
            completed=completed,config=config,padded_momentum=getattr(self,'padded_momentum',False),residual_refresh_interval=getattr(self,'residual_refresh_interval',0),stable_energy=getattr(self,'stable_energy',False)),path)

    @classmethod
    def restore(cls,path):
        s=torch.load(path,map_location='cpu',weights_only=True)
        if s['format']!='research-fp4-v1':raise ValueError('Wrong research format')
        h=cls(s['dims'],s['mode'],s['gain'],kappa=s['kappa'],beta=s['beta'],tile=s['tile'],boundary=s['boundary'])
        h.stable_energy=s.get('stable_energy',False)
        h.padded_momentum=s.get('padded_momentum',False)
        h.residual_refresh_interval=s.get('residual_refresh_interval',0)
        h.gauge.q.copy_(s['gauge_q']);h.gauge.s.copy_(s['gauge_s']);h.refresh()
        torch.cuda.set_rng_state(s['rng'])
        return h

@torch.no_grad()
def reference_dslash(h,x,parity,dagger=False):
    """Tilewise FP32 arithmetic on decoded gauge/vector; no full FP32 lattice."""
    out=Field(x.shape,'fp16')
    for sl,ids in h.tiles():
        acc=torch.zeros(1,len(ids),4,3,2,device='cuda')
        for mu in range(4):
            for back in (0,1):
                j=h.op.indices[parity,mu,back,sl].long()
                u=h.u(mu,1-parity if back else parity,j if back else ids)
                ur=torch.view_as_real(u.resolve_conj())
                v=x.get((slice(None),j)).reshape(1,len(ids),4,3,2)
                z=WilsonNormalNetwork._color(ur,v,dagger=bool(back))
                z=WilsonNormalNetwork._spin(h.projectors[(1-int(dagger)) if back else int(dagger),mu],z)
                acc+=z*h.op.signs[parity,mu,back,sl][None,:,None,None,None]
        out.put((slice(None),sl),acc.flatten(-2))
    return out

def reference_apply(h,x):
    # FP32 operations, FP16 storage between stages: independent but
    # still a quantized diagnostic. Validate against full FP32 on tiny lattice.
    def schur(z,dag):
        odd=reference_dslash(h,z,1,dag);out=reference_dslash(h,odd,0,dag)
        h.axpy(out,1.,z,-h.kappa**2,out)
        return out
    return schur(schur(x,False),True)
