"""Fresh-process operator benchmark. Loads only saved reduced-precision tensors."""
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
from packed_fp4 import PackedWilsonFP4
from reduced_forward import ReducedWilson
p=argparse.ArgumentParser();p.add_argument('--precision',required=True,choices=('fp4','fp8','fp16','bf16'))
p.add_argument('--repeats',type=int,default=30);a=p.parse_args()
result_path=Path(f'isolated_{a.precision}_result.json')
if result_path.exists():raise SystemExit('Saved measurement exists; read it instead of rerunning')
def memory():
 return {'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
         'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved()}
def tensor_bytes(t):return t.numel()*t.element_size()
with torch.no_grad():
 torch.cuda.init();baseline=memory()
 if a.precision=='fp4':
    model_path='packed_forward_56x96.pt';input_path='packed_input_56x96.pt';net=PackedWilsonFP4.load(model_path)
 else:
    model_path=f'isolated_{a.precision}_model.pt';input_path=f'isolated_{a.precision}_input.pt';net=ReducedWilson.load(model_path)
 data=torch.load(input_path,map_location='cuda',weights_only=True);q,s=data['codes'],data['scales'];del data
 compiled=torch.compile(net,fullgraph=True)
 for _ in range(3):warm=compiled(q,s)
 torch.cuda.synchronize();del warm;gc.collect();torch.cuda.empty_cache()
 torch.cuda.reset_peak_memory_stats();before_capture=memory()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=compiled(q,s)
 graph.replay();torch.cuda.synchronize();capture=memory()
 gc.collect();torch.cuda.empty_cache()
 for _ in range(5):graph.replay()
 torch.cuda.synchronize()
 torch.cuda.reset_peak_memory_stats();before_replay=memory()
 wall=[];gpu=[]
 start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 for _ in range(a.repeats):
    t=time.perf_counter();start.record();graph.replay();end.record();end.synchronize()
    wall.append(time.perf_counter()-t);gpu.append(start.elapsed_time(end)/1000)
 after_replay=memory()
 assert out[0].dtype==q.dtype and out[1].dtype==s.dtype
 assert 'libquda' not in Path('/proc/self/maps').read_text().lower()
 model_bytes=sum(tensor_bytes(t) for t in net.buffers())
 input_bytes=tensor_bytes(q)+tensor_bytes(s);output_bytes=sum(tensor_bytes(t) for t in out)
 report={'precision':a.precision,'dims':net.dims,'batch':q.shape[0],'model_file':model_path,'input_file':input_path,
         'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'repeats':a.repeats,
         'wall_samples_s':wall,'gpu_samples_s':gpu,'wall_median_s':statistics.median(wall),'gpu_median_s':statistics.median(gpu),
         'model_bytes':model_bytes,'input_bytes':input_bytes,'output_bytes':output_bytes,
         'logical_model_input_output_bytes':model_bytes+input_bytes+output_bytes,
         'baseline':baseline,'before_capture':before_capture,'capture':capture,'before_replay':before_replay,'after_replay':after_replay,
         'scope':'Operator-only Torch allocations in a fresh process. No QUDA, optimizer, master parameter, reference tensors, encoding or decoding. No whole-system RAM estimate.',
         'memory_note':'Capture peak measures transient operator allocation. Replay reserved includes graph scratch retained for execution. CUDA context/driver allocations outside Torch are not included.',
         'payload_dtype':str(q.dtype),'scale_dtype':str(s.dtype),
         'launch':{'group':net.group_sites,'unroll':net.unroll,'warps':net.num_warps}}
 result_path.write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if not k.endswith('samples_s')},indent=2),flush=True)
