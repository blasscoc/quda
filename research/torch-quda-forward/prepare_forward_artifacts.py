"""One-time export only. Not part of the isolated measurement process."""
import argparse,gc,json
from pathlib import Path
import torch
from reduced_forward import ReducedWilson
from wilson_network import load_qdp_gauge
p=argparse.ArgumentParser();p.add_argument('--gauge',required=True);a=p.parse_args()
with torch.no_grad():
 torch.manual_seed(45)
 x=torch.randn(1,56**3*96//2,4,3,2,device='cuda',dtype=torch.float64)
 gauge=load_qdp_gauge(a.gauge,(56,56,56,96),device='cuda',dtype=torch.float32)
 for dtype in ('fp16','bf16','fp8'):
    model_path=Path(f'isolated_{dtype}_model.pt');input_path=Path(f'isolated_{dtype}_input.pt')
    if model_path.exists() and input_path.exists():
        print('REUSING '+str(model_path),flush=True);continue
    net=ReducedWilson(gauge,(56,56,56,96),dtype)
    q,s=net.encode(x.float())
    expected=net(q,s)
    net.save(model_path);torch.save({'codes':q,'scales':s},input_path)
    reload=ReducedWilson.load(model_path);actual=reload(q,s)
    assert torch.equal(expected[0].view(torch.uint8),actual[0].view(torch.uint8))
    assert torch.equal(expected[1],actual[1])
    print(json.dumps({'precision':dtype,'model':str(model_path),'input':str(input_path),'roundtrip_exact':True}),flush=True)
    del net,reload,q,s,expected,actual;gc.collect();torch.cuda.empty_cache()
