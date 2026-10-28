"""Measured candidate transformations of the validated Torch forward network."""
import torch
from wilson_network import WilsonNormalNetwork, complex_matrix_as_real


class CachedWilsonNetwork(WilsonNormalNetwork):
    def __init__(self, gauge, dims, kappa=.156, temporal_boundary=1, fp4=False, fused=False, components=1):
        super().__init__(gauge,dims,kappa,temporal_boundary)
        self.fp4=fp4
        self.fused=fused
        self.components=components
        self._cache_version=-1
        self.refresh_cache()

    def save(self, path):
        raise NotImplementedError('Experimental execution modes are not serialized; save the base WilsonNormalNetwork gauge/config instead')

    @classmethod
    def load(cls, path, device='cpu'):
        base=WilsonNormalNetwork.load(path,device)
        return cls(base.gauge,base.dims,base.kappa,base.temporal_boundary)

    def refresh_cache(self):
        """Call after replacing gauge weights. Changing dtype/device uses buffers."""
        directions=[]
        for p in (0,1):
            d=[]
            for mu in range(4):
                minus=self.neighbours[p,mu,1]
                uf=complex_matrix_as_real(self.gauge[mu,p])
                ub=complex_matrix_as_real(self.gauge[mu,1-p].index_select(0,minus)).transpose(-1,-2)
                d.extend([uf,ub])
            directions.append(torch.stack(d))
        self.register_buffer('color_weights',torch.stack(directions),persistent=False)
        self._cache_version=self.gauge._version

    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse)
        self._cache_version=self.gauge._version
        return result

    def compiled_forward(self):
        """Full graph GPU computation, with an eager stale-gauge guard per call."""
        # Preserve the eager rounding at Schur subtraction boundaries: an FMA
        # can perturb an FP4 quantization decision in the following Dslash.
        compiled=torch.compile(self,fullgraph=True,options={'emulate_precision_casts':True})
        def checked(x):
            if self.gauge._version!=self._cache_version:
                raise RuntimeError('Gauge was modified: call refresh_cache() before applying the cached operator')
            return compiled(x)
        return checked

    @staticmethod
    def spin_project(x,mu,sign):
        # Sparse signed permutation gamma, equivalent to dense 8x8 projector.
        if mu==3:
            factors=x.new_tensor([1+sign,1+sign,1-sign,1-sign])
            return x*factors[None,None,:,None,None]
        cols=([3,2,1,0],[3,2,1,0],[2,3,0,1])[mu]
        z=x[:,:,cols]
        if mu in (0,2):
            z=torch.stack((-z[...,1],z[...,0]),dim=-1)
        signs=([1,1,-1,-1],[1,-1,-1,1],[1,-1,-1,1])[mu]
        s=x.new_tensor(signs)
        return x+sign*z*s[None,None,:,None,None]

    def dslash(self,x,output_parity,dagger=False):
        self._check(x)
        if not torch.compiler.is_compiling() and self.gauge._version!=self._cache_version:
            raise RuntimeError('Gauge was modified: call refresh_cache() before applying the cached operator')
        if self.fused:
            from fp4_color import fused_dslash_fp4
            return fused_dslash_fp4(self.color_weights[output_parity],self.neighbours[output_parity],
                                   self.signs[output_parity],x,dagger,self.components)
        out=torch.zeros_like(x)
        for mu in range(4):
            for direction in range(2):
                idx=self.neighbours[output_parity,mu,direction]
                z=x.index_select(1,idx)
                w=self.color_weights[output_parity,2*mu+direction]
                if self.fp4:
                    from fp4_color import color_fp4
                    z=color_fp4(w,z)
                else:
                    z=torch.matmul(z.flatten(-2),w.transpose(-1,-2)).reshape_as(z)
                sign=(1 if direction else -1)*(-1 if dagger else 1)
                z=self.spin_project(z,mu,sign)
                bc=self.signs[output_parity,mu,direction]
                out=out+z*bc[None,:,None,None,None]
        return out
