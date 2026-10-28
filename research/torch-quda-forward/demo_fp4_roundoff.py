"""Reproducible 32^4 ablations. No QUDA; diagnostics reported separately."""
import argparse,gc,json,time,statistics
from pathlib import Path
import torch
from compact_hmc import Field
from fp4_roundoff import ResearchHMC,Expansion,reference_apply

def sync():torch.cuda.synchronize();return time.perf_counter()

def seed_gauge(h,warm):
    torch.manual_seed(123)
    for sl,ids in h.tiles():
        for mu in range(4):
            for p in (0,1):
                if warm:
                    a=torch.randn(len(ids),8,device='cuda')*.05
                    g=torch.einsum('na,aij->nij',a.to(torch.complex64),h.T)
                    u=torch.linalg.matrix_exp(g)
                else:u=torch.eye(3,device='cuda',dtype=torch.complex64).expand(len(ids),3,3)
                h.gauge.put((mu,p,sl),torch.view_as_real(u).reshape(-1,3,6))
    h.refresh()

def residual(h,x,b,independent=False):
    ax=reference_apply(h,x) if independent else h.apply(x)
    # DO NOT round residual again before measuring its norm.
    rr=torch.zeros((),device='cuda');bb=torch.zeros_like(rr)
    for sl,_ in h.tiles():
        ix=(slice(None),sl);diff=b.get(ix)-ax.get(ix)
        rr+=diff.square().sum();bb+=b.get(ix).square().sum()
    return float((rr/bb).sqrt())

def solve_case(dims,mode,warm,iters,tile,gain=1.,rhs_scale=1.,restart=0):
    torch.manual_seed(45)
    h=ResearchHMC(dims,mode,gain,tile=tile)
    h.residual_refresh_interval=restart
    seed_gauge(h,warm)
    # Same unquantized RHS across modes: generated per tile then encoded.
    torch.manual_seed(71);b=h.vector()
    rhs_generator=torch.Generator(device='cuda').manual_seed(71)
    be=torch.zeros((),device='cuda');bn=torch.zeros_like(be)
    for sl,ids in h.tiles():
        ix=(slice(None),sl);raw=torch.randn(1,len(ids),4,6,device='cuda',generator=rhs_generator)*rhs_scale
        b.put(ix,raw);be+=(raw-b.get(ix)).square().sum();bn+=raw.square().sum()
    rhs_storage_error=float((be/bn).sqrt())
    # Native shape/JIT warmup excluded from solve measurement.
    warmout=h.apply(b);del warmout;gc.collect();torch.cuda.empty_cache()
    h.diagnostics=True
    torch.cuda.reset_peak_memory_stats();start=sync();x=h.solve(b,iters);elapsed=sync()-start
    peak=torch.cuda.max_memory_allocated();diagstart=sync()
    native=residual(h,x,b);oracle=residual(h,x,b,True);diagseconds=sync()-diagstart
    rows=h.solves[-1]['iterations']
    result=dict(kind='solver',dims=dims,mode=mode,warm=warm,gain=gain,rhs_scale=rhs_scale,
        iterations=iters,residual_refresh_interval=restart,rhs_storage_error=rhs_storage_error,wall_seconds=elapsed,mean_iteration_seconds=statistics.mean(r['seconds'] for r in rows),
        logged_diagnostic_seconds=sum(r['diagnostic_seconds'] for r in rows),external_diagnostic_seconds=diagseconds,
        peak_allocated_bytes=peak,vector_bytes=x.bytes(),gauge_bytes=h.gauge.bytes(),
        native_residual=native,independent_residual=oracle,history=rows)
    return result

def trajectory_case(dims,mode,tile,stable_energy=False,seed=45,padded_momentum=False):
    torch.manual_seed(seed);h=ResearchHMC(dims,mode,tile=tile);h.cold();h.diagnostics=True
    h.stable_energy=stable_energy
    h.padded_momentum=padded_momentum
    # Trace actual gauge motion, not just scalar energies or accepted state.
    drift=h.drift
    motion=[]
    def tracked(mom,dt):
        before=h.gauge.clone();drift(mom,dt)
        diff=torch.zeros((),device='cuda');unit=torch.zeros_like(diff);changed=0
        eye=torch.eye(3,device='cuda',dtype=torch.complex64)
        for sl,ids in h.tiles():
            for mu in range(4):
                for p in (0,1):
                    ix=(mu,p,sl);a=before.get(ix);b=h.gauge.get(ix)
                    changed+=int((a!=b).any(-1).any(-1).sum())
                    diff+=(a-b).square().sum()
                    u=torch.view_as_complex(b.reshape(-1,3,3,2).contiguous())
                    unit+=(u.mH@u-eye).abs().square().sum()
        motion.append(dict(changed_links=changed,total_links=8*h.v,
            rms_link_change=float((diff/(8*h.v)).sqrt()),rms_unitarity_error=float((unit/(8*h.v)).sqrt())))
    h.drift=tracked
    gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=sync()
    r=h.trajectory(1,.001,4,4);elapsed=sync()-start
    return dict(kind='trajectory',dims=dims,mode=mode,stable_energy=stable_energy,seed=seed,padded_momentum=padded_momentum,wall_seconds=elapsed,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),motion=motion,result=r,timings=h.timings,solves=h.solves)

if __name__=='__main__':
    import sys
    if any(arg == '--config' or arg.startswith('--config=') for arg in sys.argv[1:]):
        from run_hmc_config import main
        main()
        raise SystemExit(0)
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=32);p.add_argument('--iters',type=int,default=12)
    p.add_argument('--tile',type=int,default=8192);p.add_argument('--mode',default='baseline',choices=['baseline','sr','dual','center','center_dual','center_dual_mom'])
    p.add_argument('--seed',type=int,default=45)
    p.add_argument('--padded-momentum',action='store_true')
    p.add_argument('--scalar64',action='store_true')
    p.add_argument('--warm',action='store_true');p.add_argument('--trajectory',action='store_true');p.add_argument('--gain',type=float,default=1.)
    p.add_argument('--restart-every',type=int,default=0)
    p.add_argument('--rhs-scale',type=float,default=1.);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dims=[a.size]*4
    r=trajectory_case(dims,a.mode,a.tile,a.scalar64,a.seed,a.padded_momentum) if a.trajectory else solve_case(dims,a.mode,a.warm,a.iters,a.tile,a.gain,a.rhs_scale,a.restart_every)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('history','solves','timings')}),flush=True)
