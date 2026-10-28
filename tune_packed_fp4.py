"""Bounded full-size launch tuning; saves ready-to-run packed model and input."""
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
from packed_fp4 import PackedWilsonFP4,pack_vector
from wilson_network import load_qdp_gauge
p=argparse.ArgumentParser();p.add_argument('--gauge',required=True);a=p.parse_args()
path=Path('packed_launch_tuning.json')
if path.exists():raise SystemExit('Saved tuning results already exist')
torch.manual_seed(45)
with torch.no_grad():
    u=load_qdp_gauge(a.gauge,(56,56,56,96),device='cuda',dtype=torch.float32)
    net=PackedWilsonFP4(u,(56,56,56,96),group_sites=8,unroll=8);del u
    x=torch.randn(1,net.v,4,3,2,device='cuda',dtype=torch.float64)
    q,s=pack_vector(x);del x
    golden=net(q,s);torch.cuda.synchronize()
    gc.collect();torch.cuda.empty_cache()
    rows=[]
    configs=[(8,8,4)]+[(g,u,w) for w in (2,8) for g in (4,8) for u in (1,8)]
    for group,unroll,warps in configs:
        net.group_sites=group;net.unroll=unroll;net.num_warps=warps
        row=dict(group=group,unroll=unroll,warps=warps)
        try:
            result=net(q,s);torch.cuda.synchronize()
            assert torch.equal(result[0],golden[0]) and torch.equal(result[1].view(torch.uint8),golden[1].view(torch.uint8))
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=net(q,s)
            graph.replay();torch.cuda.synchronize()
            times=[]
            for _ in range(7):
                t=time.perf_counter();graph.replay();torch.cuda.synchronize();times.append(time.perf_counter()-t)
            row.update(samples_s=times,median_s=statistics.median(times),exact_output_match=True)
            del graph,out,result
        except Exception as e:
            if 'illegal' in str(e).lower():raise
            row['error']=str(e)[:1000]
        rows.append(row);path.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
    best=min((r for r in rows if 'median_s' in r),key=lambda r:r['median_s'])
    net.group_sites=best['group'];net.unroll=best['unroll'];net.num_warps=best['warps']
    net.save('packed_forward_56x96.pt')
    torch.save({'codes':q,'scales':s},'packed_input_56x96.pt')
    reloaded=PackedWilsonFP4.load('packed_forward_56x96.pt')
    result=reloaded(q,s)
    assert torch.equal(result[0],golden[0]) and torch.equal(result[1].view(torch.uint8),golden[1].view(torch.uint8))
    Path('packed_fp4.ptx').write_text(reloaded.last_kernel.asm['ptx'])
    Path('packed_best.json').write_text(json.dumps(dict(best=best,roundtrip_exact=True,model_bytes=net.storage_bytes()),indent=2))
    print('SAVED packed model and input; reload matches exactly',flush=True)
