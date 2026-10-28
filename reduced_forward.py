"""Native FP16/BF16/FP8 storage and MMA, same four-stage quantization boundaries."""
import math
import torch
import triton
import triton.language as tl
from wilson_network import checkerboard_geometry,complex_matrix_as_real

DTYPES={'fp16':torch.float16,'bf16':torch.bfloat16,'fp8':torch.float8_e4m3fn}

def encode_rows(x,precision):
    if precision!='fp8':return x.to(DTYPES[precision]).contiguous(),torch.empty(0,device=x.device)
    scale=(x.abs().amax(-1).float()/448).clamp_min(2**-9).log2().ceil().exp2()
    return (x.float()/scale[...,None]).to(torch.float8_e4m3fn).contiguous(),scale.contiguous()

def decode_rows(x,s,precision):
    return x.float()*s[...,None] if precision=='fp8' else x.float()

@triton.jit
def _reduced_dslash(W,WS,X,XS,IDX,BC,SKIP,SS,Y,YS,
                   V:tl.constexpr,B:tl.constexpr,NT:tl.constexpr,DAG:tl.constexpr,
                   SCHUR:tl.constexpr,K2:tl.constexpr,G:tl.constexpr,UNROLL:tl.constexpr,FP8:tl.constexpr):
    base=tl.program_id(0).to(tl.int64)*G
    MT:tl.constexpr=max(16,G*8)
    KT:tl.constexpr=max(32,G*8)
    m=tl.arange(0,MT);k=tl.arange(0,KT);n=tl.arange(0,NT)
    sites=base+m//8;col=m%8;spin=n%4
    ks=k//8;kc=k%8
    acc=tl.full((MT,NT),0,tl.float32)
    for d in tl.range(0,8,loop_unroll_factor=UNROLL):
        mask=(m[:,None]<G*8)&(col[:,None]<6)&(sites[:,None]<V)&(kc[None,:]<6)&((m[:,None]//8)==ks[None,:])
        source_col=tl.where(col[:,None]%2==0,kc[None,:],kc[None,:]^1)
        a=tl.load(W+(d*V+sites[:,None])*18+(col[:,None]//2)*6+source_col,mask,0.).to(tl.float32)
        a*=tl.where((col[:,None]%2==1)&(kc[None,:]%2==0),-1.,1.)
        a=a.to(W.dtype.element_ty)
        neighbor=tl.load(IDX+d*V+base+ks,(ks<G)&(base+ks<V),0)
        off=(n[None,:]//4)*V*24+neighbor[:,None]*24+spin[None,:]*6+kc[:,None]
        bp=tl.load(X+off,(ks[:,None]<G)&(base+ks[:,None]<V)&(kc[:,None]<6)&(n[None,:]<B*4),0.)
        z=tl.dot(a,bp,max_num_imprecise_acc=0)
        if FP8:
            sa=tl.load(WS+(d*V+sites)*3+col//2,(m<G*8)&(col<6)&(sites<V),1.)
            ns=tl.load(IDX+d*V+sites,(m<G*8)&(sites<V),0)
            sb=tl.load(XS+(n[None,:]//4)*V*4+ns[:,None]*4+spin[None,:],
                       (m[:,None]<G*8)&(sites[:,None]<V)&(n[None,:]<B*4),1.)
            z=z*sa[:,None]*sb
        mu=d//2
        if mu==3:
            gam=z*tl.where(spin[None,:]<2,1.,-1.)
        else:
            perm=tl.where(mu<2,n-spin+(3-spin),n^2)
            gam=tl.gather(z,tl.broadcast_to(perm[None,:],(MT,NT)),1)
            if mu==0 or mu==2:
                gam=tl.gather(gam,tl.broadcast_to((m^1)[:,None],(MT,NT)),0)
                gam=gam*tl.where((m[:,None]%2)==0,-1.,1.)
            if mu==0:gam=gam*tl.where(spin[None,:]<2,1.,-1.)
            else:gam=gam*tl.where((spin[None,:]==0)|(spin[None,:]==3),1.,-1.)
        sign=tl.where(d%2==0,-1.,1.)*(-1 if DAG else 1)
        bc=tl.load(BC+d*V+sites,(m<G*8)&(sites<V),0).to(tl.float32)
        acc+=(z+sign*gam)*bc[:,None]
    dest=(n[None,:]//4)*V*24+sites[:,None]*24+spin[None,:]*6+col[:,None]
    sdest=(n[None,:]//4)*V*4+sites[:,None]*4+spin[None,:]
    valid=(m[:,None]<G*8)&(col[:,None]<6)&(sites[:,None]<V)&(n[None,:]<B*4)
    if SCHUR:
        val=tl.load(SKIP+dest,valid,0.).to(tl.float32)
        if FP8:val*=tl.load(SS+sdest,valid,0.)
        acc=val-K2*acc
    acc=tl.where(valid,acc,0.)
    if FP8:
        group=tl.reshape(acc,(MT//8,8,NT))
        scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(group),1)/448.,2.**-9))))
        expanded=tl.reshape(tl.broadcast_to(scale[:,None,:],(MT//8,8,NT)),(MT,NT))
        acc=acc/expanded
        tl.store(YS+sdest,expanded,valid&(col[:,None]==0))
    tl.store(Y+dest,acc,valid)

class ReducedWilson(torch.nn.Module):
    def __init__(self,gauge,dims,precision,kappa=.156,temporal_boundary=1):
        super().__init__();self.dims=tuple(dims);self.v=math.prod(dims)//2
        self.precision=precision;self.kappa=kappa;self.group_sites=4;self.unroll=8;self.num_warps=1
        idx,sign=checkerboard_geometry(dims,temporal_boundary)
        self.register_buffer('indices',idx.to(device=gauge.device,dtype=torch.int32))
        self.register_buffer('signs',sign.to(device=gauge.device,dtype=torch.int8))
        self.register_buffer('weights',torch.empty(2,8,self.v,3,6,device=gauge.device,dtype=DTYPES[precision]))
        self.register_buffer('scales',torch.empty((2,8,self.v,3) if precision=='fp8' else (0,),device=gauge.device))
        with torch.no_grad():
            for p in (0,1):
                for mu in range(4):
                    for back in (0,1):
                        u=gauge[mu,p] if not back else gauge[mu,1-p].index_select(0,self.indices[p,mu,1].long())
                        w=complex_matrix_as_real(u)
                        if back:w=w.transpose(-1,-2)
                        q,s=encode_rows(w[:,::2,:],precision)
                        self.weights[p,2*mu+back].copy_(q)
                        if precision=='fp8':self.scales[p,2*mu+back].copy_(s)
    def encode(self,x):return encode_rows(x.flatten(-2),self.precision)
    def decode(self,x,s):return decode_rows(x,s,self.precision).reshape(*x.shape[:-1],3,2)
    def dslash(self,q,s,parity,dagger=False,skip=None):
        b=q.shape[0];y=torch.empty_like(q);ys=torch.empty_like(s)
        sq,ss=(q,s) if skip is None else skip
        ws=self.scales[parity] if self.precision=='fp8' else self.scales
        kernel=_reduced_dslash[(triton.cdiv(self.v,self.group_sites),)](
            self.weights[parity],ws,q,s,self.indices[parity],self.signs[parity],
            sq,ss,y,ys,self.v,b,max(16,triton.next_power_of_2(b*4)),dagger,skip is not None,
            self.kappa**2,self.group_sites,self.unroll,self.precision=='fp8',num_warps=self.num_warps,enable_fp_fusion=False)
        if not torch.compiler.is_compiling():self.last_kernel=kernel
        return y,ys
    def forward(self,q,s):
        if q.dtype!=DTYPES[self.precision] or q.shape!=(s.shape[0] if self.precision=='fp8' else q.shape[0],self.v,4,6):
            raise ValueError('Wrong input format')
        odd=self.dslash(q,s,1);middle=self.dslash(*odd,0,skip=(q,s))
        odd=self.dslash(*middle,1,True)
        return self.dslash(*odd,0,True,skip=middle)
