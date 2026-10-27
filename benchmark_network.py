"""Whole-operator timing and error, including conversion/setup separately."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch

from wilson_network import WilsonNormalNetwork, load_qdp_gauge
from optimized_network import CachedWilsonNetwork


def synchronize():
    torch.cuda.synchronize()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dims',nargs=4,type=int,default=[16,16,16,32])
    p.add_argument('--gauge')
    p.add_argument('--batches',nargs='+',type=int,default=[1,4,8])
    p.add_argument('--output',default='network_benchmark.json')
    p.add_argument('--reps',type=int,default=3)
    p.add_argument('--modes',nargs='+')
    a=p.parse_args()
    torch.manual_seed(45)
    torch.backends.cuda.matmul.allow_tf32=False
    dims=tuple(a.dims)
    v=torch.tensor(dims).prod().item()//2
    if a.gauge:
        gauge=load_qdp_gauge(a.gauge,dims)
        source=a.gauge
    else:
        # Small generated nontrivial SU(3); separate from physical performance case.
        h=torch.randn(4,2,v,3,3,dtype=torch.complex128)
        h=(h+h.mH)/2
        h=h-h.diagonal(dim1=-2,dim2=-1).sum(-1)[...,None,None]/3*torch.eye(3)
        gauge=torch.view_as_real(torch.linalg.matrix_exp(.3j*h)).contiguous()
        del h
        source='synthetic random SU3; correctness/performance smoke only'
    report={'dims':dims,'gauge_source':source,'cases':[],'torch':torch.__version__,
            'gpu':torch.cuda.get_device_name(),'fused_group_sites':2,'note':'FP4 is native color GEMM operands only, FP32 accumulation and FP32 external state; no all-FP4 claim.'}
    for batch in a.batches:
        x=torch.randn(batch,v,4,3,2,device='cuda',dtype=torch.float64)
        reference=WilsonNormalNetwork(gauge.cuda(),dims)
        with torch.no_grad():
            yref=reference(x)
        del reference
        gc.collect();torch.cuda.empty_cache()
        for mode,dtype in [('baseline_fp64',torch.float64),('baseline_fp32',torch.float32),
                           ('cached_fp32',torch.float32),('cached_fp16',torch.float16),
                           ('cached_bf16',torch.bfloat16),('native_fp4_color',torch.float32),
                           ('fused_fp4',torch.float32),('fused_fp4x3',torch.float32),('fused_fp16',torch.float16)]:
            if a.modes and mode not in a.modes:continue
            start=time.perf_counter()
            cls=WilsonNormalNetwork if mode.startswith('baseline') else CachedWilsonNetwork
            kw={'fp4':True,'fused':mode.startswith('fused'),'components':3 if mode.endswith('x3') else 1} if 'fp4' in mode else {}
            if mode=='fused_fp16':kw={'fused':True,'components':0}
            net=cls(gauge.to(device='cuda',dtype=dtype),dims,**kw)
            xx=x.to(dtype=dtype)
            synchronize();setup=time.perf_counter()-start
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                y=net(xx);synchronize()  # warmup, including JIT if needed
                samples=[]
                for _ in range(a.reps):
                    start=time.perf_counter();y=net(xx);synchronize()
                    samples.append(time.perf_counter()-start)
                err=((y.double()-yref).norm()/yref.norm()).item()
                finite=torch.isfinite(y).all().item()
            row={'batch':batch,'mode':mode,'setup_s':setup,'forward_s_samples':samples,
                 'forward_s_median':sorted(samples)[len(samples)//2],
                 'relative_error_vs_fp64':err,'finite':finite,
                 'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
            assert finite,row
            if mode=='cached_fp32':assert err<1e-5,row
            report['cases'].append(row);print(json.dumps(row),flush=True)
            Path(a.output).write_text(json.dumps(report,indent=2))
            if mode=='native_fp4_color':
                from fp4_color import color_fp4
                Path('fp4_color.ptx').write_text(color_fp4.last_kernel.asm['ptx'])
            if mode.startswith('fused_'):
                from fp4_color import fused_dslash_fp4
                Path(mode+'.ptx').write_text(fused_dslash_fp4.last_kernel.asm['ptx'])
            del net,xx,y
            gc.collect();torch.cuda.empty_cache()
        del x,yref
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
