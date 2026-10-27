"""Independent Torch quantized reference for fused native FP4 kernels."""
import json
from pathlib import Path
import torch
from optimized_network import CachedWilsonNetwork
from fp4_color import quantize_reference as q, fused_dslash_fp4


def reference(net, x, parity, dagger, components):
    b,v=x.shape[:2]
    out=torch.zeros_like(x)
    def grouped(z):
        z=z.reshape(b,v//2,2,4,6).transpose(2,3).reshape(b,v//2,4,12)
        return q(z,-1).reshape(b,v//2,4,2,6).transpose(2,3).reshape(b,v,4,6)
    for d in range(8):
        mu, direction=divmod(d,2)
        w=net.color_weights[parity,d]
        z=x.index_select(1,net.neighbours[parity,mu,direction]).flatten(-2)
        wh=q(w,-1);zh=grouped(z)
        r=zh@wh.transpose(-1,-2)
        if components==3:
            r=r+zh@q(w-wh,-1).transpose(-1,-2)+grouped(z-zh)@wh.transpose(-1,-2)
        r=net.spin_project(r.reshape_as(x),mu,(1 if direction else -1)*(-1 if dagger else 1))
        out+=r*net.signs[parity,mu,direction][None,:,None,None,None]
    return out


torch.manual_seed(81)
torch.backends.cuda.matmul.allow_tf32=False
rows=[]
with torch.no_grad():
    dims=(4,6,4,8);v=384
    h=torch.randn(4,2,v,3,3,device='cuda',dtype=torch.complex64)
    h=(h+h.mH)/2
    h-=h.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]/3*torch.eye(3,device='cuda')
    gauge=torch.view_as_real(torch.linalg.matrix_exp(.3j*h)).contiguous()
    for boundary in (1,-1):
        net=CachedWilsonNetwork(gauge,dims,temporal_boundary=boundary)
        for batch in (1,4,8):
            x=torch.randn(batch,v,4,3,2,device='cuda')
            for parity in (0,1):
                for dagger in (False,True):
                    for components in (1,3):
                        y=fused_dslash_fp4(net.color_weights[parity],net.neighbours[parity],net.signs[parity],x,dagger,components)
                        ref=reference(net,x,parity,dagger,components)
                        error=((y-ref).norm()/ref.norm()).item()
                        assert error<1e-6,(boundary,batch,parity,dagger,components,error)
                        rows.append(dict(boundary=boundary,batch=batch,parity=parity,dagger=dagger,components=components,error=error))
    net.gauge.add_(.001)
    try:
        net(x)
        raise AssertionError('stale cache accepted')
    except RuntimeError as e:
        assert 'refresh_cache' in str(e)
    net.refresh_cache()
    assert torch.isfinite(net(x)).all()
Path('fp4_validation.json').write_text(json.dumps(rows,indent=2))
print(f'{len(rows)} independent quantized-reference checks passed; max relative error {max(r["error"] for r in rows):.3g}; stale-cache guard passed')
