"""Packed FP4 in/out Wilson normal operator. No floating vector intermediates.

E2M1 nibbles plus one E4M3 power-of-two scale per six real components.
FP32 registers accumulate native FP4 MMA products and immediately requantize.
Packing/unpacking functions are explicit setup/diagnostic APIs, never called
by forward. Packed transport matrices are precomputed for both directions.
"""
import math
import torch
import triton
import triton.language as tl
from fp4_color import _e2m1, _decode
from wilson_network import checkerboard_geometry, complex_matrix_as_real


@triton.jit
def _pack6(X,Q,S,R:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    j=tl.arange(0,4)
    a=tl.load(X+r[:,None]*6+2*j[None,:],(r[:,None]<R)&(j[None,:]<3),0).to(tl.float32)
    b=tl.load(X+r[:,None]*6+2*j[None,:]+1,(r[:,None]<R)&(j[None,:]<3),0).to(tl.float32)
    scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.maximum(tl.abs(a),tl.abs(b)),1)/6.,2.**-9))))
    q=_e2m1(a/scale[:,None])|(_e2m1(b/scale[:,None])<<4)
    tl.store(Q+r[:,None]*3+j[None,:],q,(r[:,None]<R)&(j[None,:]<3))
    tl.store(S+r,scale,r<R)


@triton.jit
def _unpack6(Q,S,Y,R:tl.constexpr,BLOCK:tl.constexpr):
    r=tl.program_id(0).to(tl.int64)*BLOCK+tl.arange(0,BLOCK)
    j=tl.arange(0,4)
    q=tl.load(Q+r[:,None]*3+j[None,:],(r[:,None]<R)&(j[None,:]<3),0)
    s=tl.load(S+r,r<R,0.).to(tl.float32)
    tl.store(Y+r[:,None]*6+2*j[None,:],_decode(q&15)*s[:,None],(r[:,None]<R)&(j[None,:]<3))
    tl.store(Y+r[:,None]*6+2*j[None,:]+1,_decode(q>>4)*s[:,None],(r[:,None]<R)&(j[None,:]<3))


def pack_rows(x):
    """Setup only. Last dimension six; returns three bytes plus one FP8 scale."""
    if x.shape[-1]!=6 or not x.is_cuda:raise ValueError('Expected CUDA rows of six reals')
    x=x.contiguous();r=x.numel()//6
    q=torch.empty((*x.shape[:-1],3),device=x.device,dtype=torch.uint8)
    s=torch.empty(x.shape[:-1],device=x.device,dtype=torch.float8_e4m3fn)
    _pack6[(triton.cdiv(r,128),)](x,q,s,r,128)
    if not torch.isfinite(s.float()).all():raise ValueError('Input exceeds FP8 scale range')
    return q,s


def unpack_rows(q,s):
    """Diagnostics/export only; forward never calls this."""
    y=torch.empty((*q.shape[:-1],6),device=q.device,dtype=torch.float32)
    _unpack6[(triton.cdiv(s.numel(),128),)](q,s,y,s.numel(),128)
    return y


def pack_vector(x):return pack_rows(x.flatten(-2))
def unpack_vector(q,s):return unpack_rows(q,s).reshape(*s.shape,3,2)


@triton.jit
def _packed_dslash(W,WS,X,XS,IDX,BC,SKIP,SS,Y,YS,
                   V:tl.constexpr,B:tl.constexpr,NT:tl.constexpr,DAG:tl.constexpr,
                   SCHUR:tl.constexpr,K2:tl.constexpr,G:tl.constexpr,UNROLL:tl.constexpr):
    base=tl.program_id(0).to(tl.int64)*G
    MT:tl.constexpr=max(16,G*8)
    KT:tl.constexpr=max(32,G*8)
    SB:tl.constexpr=KT//8
    m=tl.arange(0,MT);kp=tl.arange(0,KT);n=tl.arange(0,NT)
    sites=base+m//8;col=m%8;spin=n%4
    ks=kp//8;kc=kp%8
    block=tl.arange(0,SB)
    acc=tl.full((MT,NT),0,tl.float32)
    for d in tl.range(0,8,loop_unroll_factor=UNROLL):
        mask=(m[:,None]<G*8)&(col[:,None]<6)&(sites[:,None]<V)&(kc[None,:]<3)&((m[:,None]//8)==ks[None,:])
        a=tl.load(W+(d*V+sites[:,None])*9+(col[:,None]//2)*3+kc[None,:],mask,0)
        # Store only the real rows of the complex color matrix. The imaginary
        # rows are nibble swaps/sign flips, not floating-point reconstruction.
        a=tl.where(col[:,None]%2==0,a,((a>>4)^8)|((a&15)<<4)).to(tl.uint8)
        sa=tl.load(WS+(d*V+sites)*3+col//2,(m<G*8)&(col<6)&(sites<V),1.)
        ass=tl.broadcast_to(sa[:,None],(MT,SB))
        neighbor=tl.load(IDX+d*V+base+ks,(ks<G)&(base+ks<V),0)
        off=(n[None,:]//4)*V*12+neighbor[:,None]*12+spin[None,:]*3+kc[:,None]
        bp=tl.load(X+off,(ks[:,None]<G)&(base+ks[:,None]<V)&(kc[:,None]<3)&(n[None,:]<B*4),0)
        ns=tl.load(IDX+d*V+base+block,(block<G)&(base+block<V),0)
        bss=tl.load(XS+(n[:,None]//4)*V*4+ns[None,:]*4+(n[:,None]%4),
                    (block[None,:]<G)&(base+block[None,:]<V)&(n[:,None]<B*4),1.)
        z=tl.dot_scaled(a,ass,'e2m1',bp,bss,'e2m1')
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
    dest=(n[None,:]//4)*V*12+sites[:,None]*12+spin[None,:]*3+col[:,None]//2
    sdest=(n[None,:]//4)*V*4+sites[:,None]*4+spin[None,:]
    valid=(m[:,None]<G*8)&(col[:,None]<6)&(sites[:,None]<V)&(n[None,:]<B*4)
    if SCHUR:
        old=tl.load(SKIP+dest,valid,0)
        oldscale=tl.load(SS+sdest,valid,0.).to(tl.float32)
        val=_decode(tl.where(col[:,None]%2==0,old&15,old>>4))*oldscale
        acc=val-K2*acc
    acc=tl.where(valid,acc,0.)
    group=tl.reshape(acc,(MT//8,8,NT))
    scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(group),1)/6.,2.**-9))))
    expanded=tl.reshape(tl.broadcast_to(scale[:,None,:],(MT//8,8,NT)),(MT,NT))
    code=_e2m1(acc/expanded)
    other=tl.gather(code,tl.broadcast_to((m^1)[:,None],(MT,NT)),0)
    tl.store(Y+dest,code|(other<<4),valid&(col[:,None]%2==0))
    tl.store(YS+sdest,expanded,valid&(col[:,None]==0))


class PackedWilsonFP4(torch.nn.Module):
    """Only integer payloads/indices and FP8 scale buffers persist in this model."""
    def __init__(self,gauge,dims,kappa=.156,temporal_boundary=1,group_sites=4,unroll=8,num_warps=1):
        super().__init__()
        self.dims=tuple(dims);self.kappa=kappa;self.v=math.prod(dims)//2
        self.group_sites=group_sites;self.unroll=unroll
        self.num_warps=num_warps
        if group_sites not in (1,2,4,8) or unroll not in (1,8):raise ValueError('Unsupported kernel variant')
        if num_warps not in (1,2,4,8):raise ValueError('Unsupported warp count')
        if not math.isfinite(kappa) or kappa<=0:raise ValueError('kappa must be finite and positive')
        if gauge.shape!=(4,2,self.v,3,3,2) or not gauge.is_floating_point():raise ValueError('Expected real QDP gauge layout')
        if not gauge.is_cuda:raise ValueError('Gauge must be on GPU during setup')
        idx,sign=checkerboard_geometry(dims,temporal_boundary)
        self.register_buffer('indices',idx.to(device=gauge.device,dtype=torch.int32))
        self.register_buffer('signs',sign.to(device=gauge.device,dtype=torch.int8))
        del idx,sign
        self.register_buffer('weights',torch.empty(2,8,self.v,3,3,device=gauge.device,dtype=torch.uint8))
        self.register_buffer('scales',torch.empty(2,8,self.v,3,device=gauge.device,dtype=torch.float8_e4m3fn))
        with torch.no_grad():
            for p in (0,1):
                for mu in range(4):
                    for back in (0,1):
                        u=gauge[mu,p] if not back else gauge[mu,1-p].index_select(0,self.indices[p,mu,1].long())
                        w=complex_matrix_as_real(u)
                        if back:w=w.transpose(-1,-2)
                        q,s=pack_rows(w[:,::2,:])
                        self.weights[p,2*mu+back].copy_(q);self.scales[p,2*mu+back].copy_(s)

    def dslash(self,q,s,parity,dagger=False,skip=None):
        b=q.shape[0]
        y=torch.empty_like(q);ys=torch.empty_like(s)
        sq,ss=(q,s) if skip is None else skip
        kernel=_packed_dslash[(triton.cdiv(self.v,self.group_sites),)](
            self.weights[parity],self.scales[parity],q,s,self.indices[parity],self.signs[parity],
            sq,ss,y,ys,self.v,b,max(16,triton.next_power_of_2(b*4)),dagger,skip is not None,
            self.kappa**2,self.group_sites,self.unroll,num_warps=self.num_warps,enable_fp_fusion=False)
        if not torch.compiler.is_compiling():self.last_kernel=kernel
        return y,ys

    def forward(self,q,s):
        if q.dtype!=torch.uint8 or s.dtype!=torch.float8_e4m3fn:raise ValueError('Packed FP4/FP8 inputs required')
        if q.shape!=(s.shape[0],self.v,4,3) or s.shape!=(q.shape[0],self.v,4):raise ValueError('Bad packed vector shape')
        if not q.is_contiguous() or not s.is_contiguous():raise ValueError('Packed inputs must be contiguous')
        if q.device!=self.weights.device or s.device!=q.device:raise ValueError('Packed tensors must share the model device')
        odd=self.dslash(q,s,1)
        middle=self.dslash(*odd,0,skip=(q,s))
        odd=self.dslash(*middle,1,dagger=True)
        return self.dslash(*odd,0,dagger=True,skip=middle)

    def storage_bytes(self):return sum(t.numel()*t.element_size() for t in self.buffers())

    def save(self,path):
        """Persist packed weights directly; reloading never reconstructs FP32 gauge."""
        torch.save({'format':'packed-wilson-e2m1-v1','dims':self.dims,'kappa':self.kappa,
                    'group_sites':self.group_sites,'unroll':self.unroll,'num_warps':self.num_warps,'state':self.state_dict()},path)

    @classmethod
    def load(cls,path,device='cuda'):
        data=torch.load(path,map_location=device,weights_only=True)
        if data['format']!='packed-wilson-e2m1-v1':raise ValueError('Unknown packed format')
        model=cls.__new__(cls);torch.nn.Module.__init__(model)
        model.dims=tuple(data['dims']);model.v=math.prod(model.dims)//2
        model.kappa=data['kappa'];model.group_sites=data['group_sites'];model.unroll=data['unroll']
        model.num_warps=data.get('num_warps',4)
        for name,tensor in data['state'].items():model.register_buffer(name,tensor)
        return model
