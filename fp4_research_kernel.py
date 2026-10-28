"""Experimental native FP4 Dslash: two-component vectors and I+delta links."""
import triton
import triton.language as tl
from fp4_color import _e2m1, _decode

@triton.jit
def _research_dslash(W,WS,X,XS,XL,XLS,IDX,BC,SKIP,SS,SL,SLS,Y,YS,YL,YLS,
                   V:tl.constexpr,B:tl.constexpr,NT:tl.constexpr,DAG:tl.constexpr,
                   SCHUR:tl.constexpr,K2:tl.constexpr,G:tl.constexpr,UNROLL:tl.constexpr,DUAL:tl.constexpr,CENTER:tl.constexpr):
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
        if DUAL:
            bl=tl.load(XL+off,(ks[:,None]<G)&(base+ks[:,None]<V)&(kc[:,None]<3)&(n[None,:]<B*4),0)
            bs2=tl.load(XLS+(n[:,None]//4)*V*4+ns[None,:]*4+(n[:,None]%4),
                    (block[None,:]<G)&(base+block[None,:]<V)&(n[:,None]<B*4),1.)
            z+=tl.dot_scaled(a,ass,'e2m1',bl,bs2,'e2m1')/256.
        if CENTER:
            z=z/128.
            ni=tl.load(IDX+d*V+sites,(m<G*8)&(sites<V),0)
            io=(n[None,:]//4)*V*12+ni[:,None]*12+spin[None,:]*3+col[:,None]//2
            so=(n[None,:]//4)*V*4+ni[:,None]*4+spin[None,:]
            ok=(m[:,None]<G*8)&(sites[:,None]<V)&(col[:,None]<6)&(n[None,:]<B*4)
            iq=tl.load(X+io,ok,0); isc=tl.load(XS+so,ok,0.).to(tl.float32)
            z+=_decode(tl.where(col[:,None]%2==0,iq&15,iq>>4))*isc
            if DUAL:
                iq2=tl.load(XL+io,ok,0); isc2=tl.load(XLS+so,ok,0.).to(tl.float32)
                z+=_decode(tl.where(col[:,None]%2==0,iq2&15,iq2>>4))*isc2/256.
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
        if DUAL:
            lo=tl.load(SL+dest,valid,0)
            los=tl.load(SLS+sdest,valid,0.).to(tl.float32)
            val+=_decode(tl.where(col[:,None]%2==0,lo&15,lo>>4))*los/256.
        acc=val-K2*acc
    acc=tl.where(valid,acc,0.)
    group=tl.reshape(acc,(MT//8,8,NT))
    scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(group),1)/6.,2.**-9))))
    expanded=tl.reshape(tl.broadcast_to(scale[:,None,:],(MT//8,8,NT)),(MT,NT))
    code=_e2m1(acc/expanded)
    other=tl.gather(code,tl.broadcast_to((m^1)[:,None],(MT,NT)),0)
    tl.store(Y+dest,code|(other<<4),valid&(col[:,None]%2==0))
    tl.store(YS+sdest,expanded,valid&(col[:,None]==0))

    if DUAL:
        residual=(acc-_decode(code)*expanded)*256.
        rg=tl.reshape(residual,(MT//8,8,NT))
        ls=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(rg),1)/6.,2.**-9))))
        le=tl.reshape(tl.broadcast_to(ls[:,None,:],(MT//8,8,NT)),(MT,NT))
        lc=_e2m1(residual/le)
        other2=tl.gather(lc,tl.broadcast_to((m^1)[:,None],(MT,NT)),0)
        tl.store(YL+dest,lc|(other2<<4),valid&(col[:,None]%2==0))
        tl.store(YLS+sdest,le,valid&(col[:,None]==0))
