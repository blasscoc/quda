"""Time the packed-in/packed-out forward only; conversion is separate."""
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
from packed_fp4 import PackedWilsonFP4,pack_vector,unpack_vector
from wilson_network import WilsonNormalNetwork,load_qdp_gauge

p=argparse.ArgumentParser();p.add_argument('--gauge',required=True)
p.add_argument('--dims',nargs=4,type=int,default=[56,56,56,96])
p.add_argument('--batches',nargs='+',type=int,default=[1,4])
p.add_argument('--output',default='packed_56x96_benchmark.json');a=p.parse_args()
if Path(a.output).exists():raise SystemExit('Saved result already exists; use another --output to remeasure')
torch.manual_seed(45);torch.backends.cuda.matmul.allow_tf32=False
def sync():torch.cuda.synchronize()
def clock(fn,reps=5):
    vals=[]
    for _ in range(reps):
        t=time.perf_counter();result=fn();sync();vals.append(time.perf_counter()-t)
    return result,vals
def size(qs):return sum(t.numel()*t.element_size() for t in qs)
gauge=load_qdp_gauge(a.gauge,tuple(a.dims))
t=time.perf_counter()
net=PackedWilsonFP4(gauge.to(device='cuda',dtype=torch.float32),tuple(a.dims))
sync();setup=time.perf_counter()-t
gc.collect();torch.cuda.empty_cache()
report={'dims':a.dims,'gauge':a.gauge,'seed':45,'gauge_setup_s':setup,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'storage_bytes':{n:t.numel()*t.element_size() for n,t in net.named_buffers()},
        'storage_dtypes':{n:str(t.dtype) for n,t in net.named_buffers()},
        'contract':'FP4 payloads + FP8 scales in/out and between all stages; FP32 registers only; no FP64 in forward',
        'cases':[]}
def save():Path(a.output).write_text(json.dumps(report,indent=2))
with torch.no_grad():
 for batch in a.batches:
    x=torch.randn(batch,net.v,4,3,2,device='cuda',dtype=torch.float64)
    reference=WilsonNormalNetwork(gauge.cuda(),tuple(a.dims))
    expected=reference(x);sync();del reference
    t=time.perf_counter();q,s=pack_vector(x);sync();pack_s=time.perf_counter()-t
    del x;gc.collect();torch.cuda.empty_cache()
    first=None
    for group,unroll in [(1,1),(2,1),(4,1),(8,1),(2,8),(4,8),(8,8)]:
        net.group_sites=group;net.unroll=unroll
        t=time.perf_counter();eager=net(q,s);sync();warm=time.perf_counter()-t
        # Check output contract and semantic invariance across scheduling variants.
        assert eager[0].dtype==torch.uint8 and eager[1].dtype==torch.float8_e4m3fn
        assert torch.isfinite(eager[1].float()).all()
        if first is None:first=(eager[0].clone(),eager[1].clone())
        else:
            assert torch.equal(eager[0],first[0])
            assert torch.equal(eager[1].view(torch.uint8),first[1].view(torch.uint8))
        out,plain=clock(lambda:net(q,s))
        # Capture four native kernels once. Replay neither converts nor allocates
        # vector data; graph output buffers are overwritten on each replay.
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=net(q,s)
        graph.replay();sync()
        assert torch.equal(captured[0],eager[0])
        assert torch.equal(captured[1].view(torch.uint8),eager[1].view(torch.uint8))
        _,replays=clock(graph.replay)
        decoded=unpack_vector(*captured)
        error=((decoded.double()-expected).norm()/expected.norm()).item()
        row={'batch':batch,'sites_per_program':group,'loop_unroll':unroll,'warmup_compile_s':warm,
             'input_pack_s':pack_s,'packed_input_bytes':size((q,s)),
             'direct_samples_s':plain,'direct_median_s':statistics.median(plain),
             'graph_samples_s':replays,'graph_median_s':statistics.median(replays),
             'relative_error_vs_fp64':error,'matches_other_schedules':True}
        report['cases'].append(row);save();print(json.dumps(row),flush=True)
        if group==8 and unroll==1:Path('packed_fp4.ptx').write_text(net.last_kernel.asm['ptx'])
        del decoded,graph,captured,eager,out
    best=min((r for r in report['cases'] if r['batch']==batch),key=lambda r:r['graph_median_s'])
    net.group_sites=best['sites_per_program'];net.unroll=best['loop_unroll']
    compiled=torch.compile(net,fullgraph=True)
    t=time.perf_counter();co=compiled(q,s);sync();compile_s=time.perf_counter()-t
    assert torch.equal(co[0],first[0]) and torch.equal(co[1].view(torch.uint8),first[1].view(torch.uint8))
    _,compiled_samples=clock(lambda:compiled(q,s))
    best['torch_compile_warmup_s']=compile_s
    best['torch_compile_samples_s']=compiled_samples
    best['torch_compile_median_s']=statistics.median(compiled_samples)
    del compiled,co
    save()
    del q,s,expected,first;gc.collect();torch.cuda.empty_cache()
report['best_by_batch']={str(b):min((r for r in report['cases'] if r['batch']==b),key=lambda r:r['graph_median_s']) for b in a.batches}
report['completed']=True;save()
