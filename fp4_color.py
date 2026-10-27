"""Experimental site-dependent color contraction using Triton's native FP4 dot.

Actual E2M1 operands + FP8 power-of-two scales, FP32 hardware accumulation.
This initial experiment pads a 6x6 real color matrix to the MMA tile. It does
not claim good tile utilization. No gradient is defined through quantization.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(v):
    a = tl.abs(v)
    c = tl.where(a <= .25, 0, tl.where(a < .75, 1, tl.where(a <= 1.25, 2,
        tl.where(a < 1.75, 3, tl.where(a <= 2.5, 4, tl.where(a < 3.5, 5,
        tl.where(a <= 5., 6, 7)))))))
    return (c | tl.where(v < 0, 8, 0)).to(tl.uint8)


@triton.jit
def _decode(c):
    i=c&7
    a=tl.where(i<4,i.to(tl.float32)*.5,tl.where(i==4,2.,tl.where(i==5,3.,tl.where(i==6,4.,6.))))
    return tl.where((c&8)!=0,-a,a)


@triton.jit
def _color_fp4(W, X, Y, V: tl.constexpr, B: tl.constexpr, NT: tl.constexpr):
    site = tl.program_id(0)
    m = tl.arange(0, 16)
    kp = tl.arange(0, 32)
    n = tl.arange(0, NT)
    a0 = tl.load(W+site*36+m[:,None]*6+2*kp[None,:], (m[:,None]<6)&(2*kp[None,:]<6),0).to(tl.float32)
    a1 = tl.load(W+site*36+m[:,None]*6+2*kp[None,:]+1, (m[:,None]<6)&(2*kp[None,:]+1<6),0).to(tl.float32)
    sa = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.maximum(tl.abs(a0),tl.abs(a1)),1)/6., 2.**-9))))
    ap = _e2m1(a0/sa[:,None]) | (_e2m1(a1/sa[:,None]) << 4)
    off = (n[None,:]//4)*V*24 + site*24 + (n[None,:]%4)*6 + 2*kp[:,None]
    b0 = tl.load(X+off, (n[None,:]<B*4)&(2*kp[:,None]<6),0).to(tl.float32)
    b1 = tl.load(X+off+1, (n[None,:]<B*4)&(2*kp[:,None]+1<6),0).to(tl.float32)
    sb = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.maximum(tl.abs(b0),tl.abs(b1)),0)/6., 2.**-9))))
    bp = _e2m1(b0/sb[None,:]) | (_e2m1(b1/sb[None,:]) << 4)
    ass = tl.broadcast_to(sa[:,None],(16,4)).to(tl.float8e4nv)
    bss = tl.broadcast_to(sb[:,None],(NT,4)).to(tl.float8e4nv)
    out = tl.dot_scaled(ap, ass, 'e2m1', bp, bss, 'e2m1')
    dest = (n[None,:]//4)*V*24 + site*24 + (n[None,:]%4)*6 + m[:,None]
    tl.store(Y+dest,out,(m[:,None]<6)&(n[None,:]<B*4))


def color_fp4(w, x):
    """w[V,6,6] column-vector weights; x[B,V,4,3,2]."""
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError('Experimental FP4 contraction is forward-only; no silent fake gradient')
    x = x.contiguous()
    w = w.contiguous()
    b,v = x.shape[:2]
    y = torch.empty_like(x)
    color_fp4.last_kernel = _color_fp4[(v,)](w,x,y,v,b,max(16,triton.next_power_of_2(b*4)),num_warps=4)
    return y


def quantize_reference(x, dim):
    """Independent Torch quantize/dequantize reference for kernel validation."""
    scale = torch.pow(2., torch.ceil(torch.log2(x.abs().amax(dim=dim,keepdim=True).float().div(6).clamp_min(2**-9))))
    z = x.float()/scale
    levels = torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device=x.device)
    distances = (z.abs().unsqueeze(-1)-levels).abs()
    nearest = distances == distances.amin(-1,keepdim=True)
    # Resolve halfway cases to an even E2M1 code, independently of the kernel.
    ranks = torch.tensor([0,9,2,11,4,13,6,15],device=x.device)
    idx = torch.where(nearest,ranks,100).argmin(-1)
    return levels[idx]*z.sign()*scale


@triton.jit
def _fused_dslash_fp4(W,X,IDX,SIGNS,Y,V:tl.constexpr,B:tl.constexpr,NT:tl.constexpr,DAG:tl.constexpr,COMP:tl.constexpr,G:tl.constexpr):
    # Eight direction caches exceed signed 32-bit element offsets at 56^3x96.
    base=tl.program_id(0).to(tl.int64)*G
    m=tl.arange(0,16)
    kp=tl.arange(0,8 if COMP==0 else 32)
    n=tl.arange(0,NT)
    spin=n%4
    acc=tl.full((16,NT),0,tl.float32)
    sites=base+m//6
    input_sites=base+(2*kp)//6
    for d in tl.static_range(8):
        neighbor=tl.load(IDX+d*V+input_sites,(input_sites<V)&(2*kp<G*6),0)
        woff=(d*V+sites[:,None])*36+(m[:,None]%6)*6+(2*kp[None,:])%6
        mask=(m[:,None]<G*6)&(sites[:,None]<V)&(2*kp[None,:]<G*6)&((m[:,None]//6)==((2*kp[None,:])//6))
        a0=tl.load(W+woff,mask,0).to(tl.float32)
        a1=tl.load(W+woff+1,mask,0).to(tl.float32)
        sa=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.maximum(tl.abs(a0),tl.abs(a1)),1)/6.,2.**-9))))
        ap=_e2m1(a0/sa[:,None])|(_e2m1(a1/sa[:,None])<<4)
        off=(n[None,:]//4)*V*24+neighbor[:,None]*24+(n[None,:]%4)*6+(2*kp[:,None])%6
        bmask=(n[None,:]<B*4)&(2*kp[:,None]<G*6)&(input_sites[:,None]<V)
        b0=tl.load(X+off,bmask,0).to(tl.float32)
        b1=tl.load(X+off+1,bmask,0).to(tl.float32)
        sb=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.maximum(tl.abs(b0),tl.abs(b1)),0)/6.,2.**-9))))
        bp=_e2m1(b0/sb[None,:])|(_e2m1(b1/sb[None,:])<<4)
        if COMP==0:
            # Fair fusion control: native FP16 MMA with minimal K=16 padding.
            aa=tl.interleave(a0,a1).to(tl.float16)
            bb=tl.interleave(tl.trans(b0),tl.trans(b1)).T.to(tl.float16)
            z=tl.dot(aa,bb)
        else:
            ass=tl.broadcast_to(sa[:,None],(16,4)).to(tl.float8e4nv)
            bss=tl.broadcast_to(sb[:,None],(NT,4)).to(tl.float8e4nv)
            z=tl.dot_scaled(ap,ass,'e2m1',bp,bss,'e2m1')
        if COMP==3:
            al0=a0-_decode(ap&15)*sa[:,None]
            al1=a1-_decode(ap>>4)*sa[:,None]
            bl0=b0-_decode(bp&15)*sb[None,:]
            bl1=b1-_decode(bp>>4)*sb[None,:]
            sal=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.maximum(tl.abs(al0),tl.abs(al1)),1)/6.,2.**-9))))
            sbl=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.maximum(tl.abs(bl0),tl.abs(bl1)),0)/6.,2.**-9))))
            alp=_e2m1(al0/sal[:,None])|(_e2m1(al1/sal[:,None])<<4)
            blp=_e2m1(bl0/sbl[None,:])|(_e2m1(bl1/sbl[None,:])<<4)
            als=tl.broadcast_to(sal[:,None],(16,4)).to(tl.float8e4nv)
            bls=tl.broadcast_to(sbl[:,None],(NT,4)).to(tl.float8e4nv)
            z=tl.dot_scaled(alp,als,'e2m1',bp,bss,'e2m1',acc=z)
            z=tl.dot_scaled(ap,ass,'e2m1',blp,bls,'e2m1',acc=z)
        if d//2==3:
            gam=z*tl.where(spin[None,:]<2,1.,-1.)
        else:
            if d//2<2:
                perm=n-spin+(3-spin)
            else:
                perm=n^2
            gam=tl.gather(z,tl.broadcast_to(perm[None,:],(16,NT)),1)
            if d//2==0 or d//2==2:
                gam=tl.gather(gam,tl.broadcast_to((m^1)[:,None],(16,NT)),0)
                gam=gam*tl.where((m[:,None]%2)==0,-1.,1.)
            if d//2==0:
                gam=gam*tl.where(spin[None,:]<2,1.,-1.)
            else:
                gam=gam*tl.where((spin[None,:]==0)|(spin[None,:]==3),1.,-1.)
        sign=(1 if d%2 else -1)*(-1 if DAG else 1)
        bc=tl.load(SIGNS+d*V+sites,(sites<V)&(m<G*6),0)
        acc+=(z+sign*gam)*bc[:,None]
    dest=(n[None,:]//4)*V*24+sites[:,None]*24+(n[None,:]%4)*6+m[:,None]%6
    tl.store(Y+dest,acc,(m[:,None]<G*6)&(sites[:,None]<V)&(n[None,:]<B*4))


def fused_dslash_fp4(weights,indices,signs,x,dagger=False,components=1,group_sites=2):
    if components not in (0,1,3):
        raise ValueError('components must be 0 (FP16 control), 1 or 3')
    if group_sites not in (1,2):raise ValueError('group_sites must be 1 or 2')
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError('FP4 forward experiment has no gradient through quantization')
    x=x.contiguous()
    b,v=x.shape[:2]
    y=torch.empty_like(x)
    fused_dslash_fp4.last_kernel=_fused_dslash_fp4[(triton.cdiv(v,group_sites),)](weights,x,indices,signs,y,v,b,
        max(16,triton.next_power_of_2(b*4)),dagger,components,group_sites,num_warps=4)
    return y


if __name__ == '__main__':
    import json
    torch.manual_seed(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows=[]
    for b in [1,4,8]:
        w=torch.randn(1024,6,6,device='cuda',dtype=torch.float32)*.3
        x=torch.randn(b,1024,4,3,2,device='cuda',dtype=torch.float32)
        y=color_fp4(w,x)
        qw=quantize_reference(w,-1)
        qx=quantize_reference(x.flatten(-2),-1)
        ref=torch.matmul(qx,qw.transpose(-1,-2)).reshape_as(x)
        err=((y-ref).norm()/ref.norm()).item()
        assert err<1e-6,err
        rows.append({'batch':b,'relative_error_vs_quantized_reference':err})
    print(json.dumps(rows),flush=True)
    from pathlib import Path
    Path('fp4_color.ptx').write_text(color_fp4.last_kernel.asm['ptx'])
