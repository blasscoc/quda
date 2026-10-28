"""Final warp-count sweep using only saved FP4 model/input; no conversion."""
import json,statistics,time
from pathlib import Path
import torch
from packed_fp4 import PackedWilsonFP4
path=Path('packed_final_tuning.json')
if path.exists():raise SystemExit('Saved final tuning results already exist')
with torch.no_grad():
 net=PackedWilsonFP4.load('packed_forward_56x96.pt')
 data=torch.load('packed_input_56x96.pt',map_location='cuda',weights_only=True)
 q,s=data['codes'],data['scales'];gold=net(q,s);torch.cuda.synchronize()
 original=(net.group_sites,net.unroll,net.num_warps)
 rows=[]
 for group,unroll,warps in [original]+[(g,u,1) for g in (2,4,8) for u in (1,8)]:
    row=dict(group=group,unroll=unroll,warps=warps)
    net.group_sites=group;net.unroll=unroll;net.num_warps=warps
    try:
        out=net(q,s);torch.cuda.synchronize()
        assert torch.equal(out[0],gold[0]) and torch.equal(out[1].view(torch.uint8),gold[1].view(torch.uint8))
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=net(q,s)
        graph.replay();torch.cuda.synchronize()
        samples=[]
        for _ in range(9):
            t=time.perf_counter();graph.replay();torch.cuda.synchronize();samples.append(time.perf_counter()-t)
        row.update(samples_s=samples,median_s=statistics.median(samples),exact_output_match=True)
        del graph,out
    except Exception as e:
        if 'illegal' in str(e).lower():raise
        row['error']=str(e)[:800]
    rows.append(row);path.write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 best=min((r for r in rows if 'median_s' in r),key=lambda r:r['median_s'])
 net.group_sites=best['group'];net.unroll=best['unroll'];net.num_warps=best['warps']
 compiled=torch.compile(net,fullgraph=True)
 out=compiled(q,s);torch.cuda.synchronize()
 assert torch.equal(out[0],gold[0]) and torch.equal(out[1].view(torch.uint8),gold[1].view(torch.uint8))
 samples=[]
 for _ in range(9):
    t=time.perf_counter();out=compiled(q,s);torch.cuda.synchronize();samples.append(time.perf_counter()-t)
 best['compiled_samples_s']=samples;best['compiled_median_s']=statistics.median(samples)
 net.save('packed_forward_56x96.pt')
 restored=PackedWilsonFP4.load('packed_forward_56x96.pt');out=restored(q,s)
 assert torch.equal(out[0],gold[0]) and torch.equal(out[1].view(torch.uint8),gold[1].view(torch.uint8))
 Path('packed_fp4.ptx').write_text(restored.last_kernel.asm['ptx'])
 Path('packed_best.json').write_text(json.dumps(dict(best=best,roundtrip_exact=True,model_bytes=net.storage_bytes()),indent=2))
 print('FINAL '+json.dumps(best),flush=True)
