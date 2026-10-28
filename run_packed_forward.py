"""Standalone saved FP4 input -> FP4 output. No gauge/input conversion."""
import argparse,json,statistics,time
import torch
from packed_fp4 import PackedWilsonFP4
p=argparse.ArgumentParser()
p.add_argument('--model',default='packed_forward_56x96.pt')
p.add_argument('--input',default='packed_input_56x96.pt')
p.add_argument('--output',default='packed_output_56x96.pt')
p.add_argument('--repeats',type=int,default=10)
a=p.parse_args()
if a.repeats<1:p.error('--repeats must be positive')
with torch.no_grad():
 net=PackedWilsonFP4.load(a.model)
 data=torch.load(a.input,map_location='cuda',weights_only=True)
 q,s=data['codes'],data['scales']
 compiled=torch.compile(net,fullgraph=True)
 compiled(q,s);torch.cuda.synchronize()
 graph=torch.cuda.CUDAGraph()
 with torch.cuda.graph(graph):out=compiled(q,s)
 graph.replay();torch.cuda.synchronize()
 times=[]
 for _ in range(a.repeats):
    t=time.perf_counter();graph.replay();torch.cuda.synchronize();times.append(time.perf_counter()-t)
 assert out[0].dtype==torch.uint8 and out[1].dtype==torch.float8_e4m3fn
 torch.save({'codes':out[0],'scales':out[1]},a.output)
 result={'batch':q.shape[0],'dims':net.dims,'samples_s':times,'median_s':statistics.median(times),
         'output':a.output,'output_payload_dtype':str(out[0].dtype),'output_scale_dtype':str(out[1].dtype),
         'group_sites':net.group_sites,'unroll':net.unroll,'num_warps':net.num_warps,
         'note':'Saved FP4 model/input; no packing/unpacking. Compilation and disk IO excluded from timing.'}
 print(json.dumps(result,indent=2))
