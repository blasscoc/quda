"""Independent dequantized reference for the deliberately all-packed algorithm."""
import json
from pathlib import Path
import torch
from packed_fp4 import PackedWilsonFP4,pack_vector,unpack_vector,unpack_rows
from optimized_network import CachedWilsonNetwork
from fp4_color import quantize_reference as quant
from wilson_network import WilsonNormalNetwork

torch.manual_seed(63);torch.backends.cuda.matmul.allow_tf32=False
dims=(4,6,4,8);v=384
h=torch.randn(4,2,v,3,3,device='cuda',dtype=torch.complex64)
h=(h+h.mH)/2
h-=h.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]/3*torch.eye(3,device='cuda')
u=torch.view_as_real(torch.linalg.matrix_exp(.3j*h)).contiguous()
rows=[]
with torch.no_grad():
 for boundary in (1,-1):
  for group in (1,2,4,8):
   for unroll in (1,8):
    net=PackedWilsonFP4(u,dims,temporal_boundary=boundary,group_sites=group,unroll=unroll)
    def ref(x,p,dag=False,skip=None):
        out=torch.zeros_like(x)
        for mu in range(4):
            for back in range(2):
                even=unpack_rows(net.weights[p,2*mu+back],net.scales[p,2*mu+back])
                pair=even.reshape(v,3,3,2)
                odd=torch.stack((-pair[...,1],pair[...,0]),-1).flatten(-2)
                w=torch.stack((even,odd),-2).reshape(v,6,6)
                z=x.index_select(1,net.indices[p,mu,back].long()).flatten(-2)@w.transpose(-1,-2)
                z=CachedWilsonNetwork.spin_project(z.reshape_as(x),mu,(1 if back else -1)*(-1 if dag else 1))
                out+=z*net.signs[p,mu,back][None,:,None,None,None]
        if skip is not None:out=skip-net.kappa**2*out
        return quant(out.flatten(-2),-1).reshape_as(out)
    for b in (1,4):
        x=torch.randn(b,v,4,3,2,device='cuda')
        q,s=pack_vector(x);qx=unpack_vector(q,s)
        assert torch.equal(qx,quant(x.flatten(-2),-1).reshape_as(x))
        for p in (0,1):
         for dag in (False,True):
          out=unpack_vector(*net.dslash(q,s,p,dag))
          expected=ref(qx,p,dag)
          err=((out-expected).norm()/expected.norm()).item()
          assert err<1e-6,(boundary,group,unroll,b,p,dag,err)
        mid=ref(ref(qx,1),0,skip=qx)
        expected=ref(ref(mid,1,True),0,True,skip=mid)
        result=net(q,s);out=unpack_vector(*result)
        err=((out-expected).norm()/expected.norm()).item()
        assert err<1e-6,(boundary,group,unroll,b,err)
        assert all(torch.isfinite(t.float()).all() for t in result)
        oracle=WilsonNormalNetwork(u.double(),dims,temporal_boundary=boundary)(x.double())
        physical=((out.double()-oracle).norm()/oracle.norm()).item()
        rows.append(dict(boundary=boundary,group=group,unroll=unroll,batch=b,reference_error=err,physical_error=physical))
        print(rows[-1],flush=True)
Path('packed_validation.json').write_text(json.dumps(rows,indent=2))
Path('packed_fp4.ptx').write_text(net.last_kernel.asm['ptx'])
print('PASSED all 128 Dslash / 32 full-operator checks and packing checks',flush=True)
