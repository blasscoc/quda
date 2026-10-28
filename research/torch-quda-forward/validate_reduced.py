"""Independent float-reference checks at identical reduced-storage boundaries."""
import json
from pathlib import Path
import torch
from reduced_forward import ReducedWilson,encode_rows,decode_rows
from optimized_network import CachedWilsonNetwork
from wilson_network import WilsonNormalNetwork
torch.manual_seed(63);torch.backends.cuda.matmul.allow_tf32=False
dims=(4,6,4,8);v=384
h=torch.randn(4,2,v,3,3,device='cuda',dtype=torch.complex64);h=(h+h.mH)/2
h-=h.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]/3*torch.eye(3,device='cuda')
u=torch.view_as_real(torch.linalg.matrix_exp(.3j*h)).contiguous();rows=[]
with torch.no_grad():
 for precision in ('fp16','bf16','fp8'):
  for boundary in (1,-1):
   net=ReducedWilson(u,dims,precision,temporal_boundary=boundary)
   def rounded(x):return net.decode(*net.encode(x))
   def ref(x,p,dag=False,skip=None):
    out=torch.zeros_like(x)
    for mu in range(4):
     for back in range(2):
      scale=net.scales[p,2*mu+back] if precision=='fp8' else net.scales
      even=decode_rows(net.weights[p,2*mu+back],scale,precision)
      pair=even.reshape(v,3,3,2);odd=torch.stack((-pair[...,1],pair[...,0]),-1).flatten(-2)
      w=torch.stack((even,odd),-2).reshape(v,6,6)
      z=x.index_select(1,net.indices[p,mu,back].long()).flatten(-2)@w.transpose(-1,-2)
      z=CachedWilsonNetwork.spin_project(z.reshape_as(x),mu,(1 if back else -1)*(-1 if dag else 1))
      out+=z*net.signs[p,mu,back][None,:,None,None,None]
    if skip is not None:out=skip-net.kappa**2*out
    return rounded(out)
   for batch in (1,4):
    x=torch.randn(batch,v,4,3,2,device='cuda');q,s=net.encode(x);qx=net.decode(q,s)
    for parity in (0,1):
     for dag in (False,True):
      out=net.decode(*net.dslash(q,s,parity,dag));expected=ref(qx,parity,dag)
      err=((out-expected).norm()/expected.norm()).item()
      assert err<{'fp16':1e-4,'bf16':5e-4,'fp8':1e-3}[precision],(precision,parity,dag,err)
    mid=ref(ref(qx,1),0,skip=qx);expected=ref(ref(mid,1,True),0,True,skip=mid)
    out=net.decode(*net(q,s));err=((out-expected).norm()/expected.norm()).item()
    assert err<{'fp16':1e-4,'bf16':5e-4,'fp8':1e-3}[precision],(precision,err)
    oracle=WilsonNormalNetwork(u.double(),dims,temporal_boundary=boundary)(x.double())
    physical=((out.double()-oracle).norm()/oracle.norm()).item()
    assert physical<{'fp16':.002,'bf16':.02,'fp8':.2}[precision],physical
    compiled=torch.compile(net,fullgraph=True);co=net.decode(*compiled(q,s))
    assert torch.equal(out,co)
    row=dict(precision=precision,boundary=boundary,batch=batch,reference_error=err,fp64_error=physical,compiled_exact=True)
    rows.append(row);print(row,flush=True)
   Path(precision+'_forward.ptx').write_text(net.last_kernel.asm['ptx'])
Path('reduced_validation.json').write_text(json.dumps(rows,indent=2))
print('PASS: 48 Dslash checks, 12 full-forward checks, compiled agreement',flush=True)
