"""CG FP64 and compiled FP4 approximate-gradient SGD: time to true residual."""
import argparse, ctypes, gc, json, os, statistics, time
from pathlib import Path
import torch
from wilson_network import load_qdp_gauge, to_quda_native
from optimized_network import CachedWilsonNetwork

p=argparse.ArgumentParser()
p.add_argument('--bridge',required=True);p.add_argument('--gauge',required=True)
p.add_argument('--steps',type=int,default=300)
p.add_argument('--output',default='solver_tolerances.json')
p.add_argument('--resume-cg',action='store_true',help='Reuse completed CG rows, restart the SGD trial')
a=p.parse_args()
dest=Path(a.output)
previous=json.loads(dest.read_text()) if dest.exists() and a.resume_cg else None
if dest.exists() and not a.resume_cg:
    raise SystemExit(f'Result already exists: {dest}. Read it or explicitly choose a new --output; no automatic rerun.')
os.environ.setdefault('QUDA_RESOURCE_PATH',str(Path('quda-cache').resolve()))
assert (Path(os.environ['QUDA_RESOURCE_PATH'])/'tunecache.tsv').exists(),'Missing persistent tuning cache'
report={'dims':[56,56,56,96],'gauge':a.gauge,'seed':45,'kappa':.156,
        'targets':[1e-2,1e-4,1e-8],'max_sgd_steps':a.steps,'completed':False,
        'boundary':'periodic','initial_guess':'zero','batch':1,'cg':[],'sgd':{},
        'residual_definition':'FP64 QUDA ||b-Ax||/||b||, checked independently of FP4',
        'timing_definition':'setup/warmup excluded; update-only and validated wall time reported separately',
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}
if previous:
    for key in ('dims','gauge','seed','kappa','targets','boundary','initial_guess','batch','torch','gpu'):
        assert previous[key]==report[key],f'Resume configuration mismatch: {key}'
    report['cg']=previous['cg']
    report['cg_reused_from_previous_phase']=True
def save():
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(dest)
def sync():torch.cuda.synchronize()
def ptr(t):return ctypes.c_void_p(t.data_ptr())
lib=ctypes.CDLL(a.bridge);I=ctypes.c_int;D=ctypes.c_double;P=ctypes.c_void_p
lib.quda_replay_init.argtypes=[I]*4+[ctypes.c_char_p,D,D,ctypes.c_ulonglong]
lib.quda_replay_init.restype=I
lib.quda_replay_apply_mdagm_d2d.argtypes=[P,P];lib.quda_replay_apply_mdagm_d2d.restype=None
lib.quda_replay_solve_cg_d2d.argtypes=[P,P,D,I,ctypes.POINTER(I)]+[ctypes.POINTER(D)]*3
lib.quda_replay_solve_cg_d2d.restype=None
lib.quda_replay_finalize.restype=None
lib.flushPoolQuda.argtypes=[I];lib.flushPoolQuda.restype=None
t=time.perf_counter()
rc=lib.quda_replay_init(56,56,56,96,a.gauge.encode(),5.6,.156,20461026)
if rc:raise RuntimeError(rc)
report['quda_setup_s']=time.perf_counter()-t;save()
try:
    torch.manual_seed(45)
    v=56**3*96//2
    bsite=torch.randn(1,v,4,3,2,device='cuda',dtype=torch.float64)
    b=to_quda_native(bsite);bnorm=b.norm()
    ax=torch.empty_like(b);sol=torch.zeros_like(b)
    def apply(x,y):lib.quda_replay_apply_mdagm_d2d(ptr(x),ptr(y));sync()
    def residual_native(x):
        apply(x,ax);ax.sub_(b)
        return (ax.norm()/bnorm).item()
    apply(b,ax)
    for tol in report['targets']:
        if any(r['target']==tol for r in report['cg']):continue
        sol.zero_();sync()
        it=I();res=D();seconds=D();flops=D()
        start=time.perf_counter()
        lib.quda_replay_solve_cg_d2d(ptr(b),ptr(sol),tol,1000,
            ctypes.byref(it),ctypes.byref(res),ctypes.byref(seconds),ctypes.byref(flops))
        sync();solve_s=time.perf_counter()-start
        checked=residual_native(sol);wall=time.perf_counter()-start
        row={'target':tol,'iterations':it.value,'solve_s':solve_s,
             'validated_wall_s':wall,'internal_s':seconds.value,
             'mean_internal_s_per_iteration':seconds.value/max(it.value,1),
             'true_residual':checked,'reached':checked<=tol,
             'step_timing_note':'internal solver runtime / iterations, includes solver overhead; not individual kernel timing'}
        report['cg'].append(row);save();print('CG '+json.dumps(row),flush=True)
    # Calibrate plain SGD on this exact operator, not an unrelated learned model.
    # Power iteration is an estimate, not a proven upper spectral bound.
    start=time.perf_counter();sol.copy_(b/bnorm)
    for _ in range(20):
        apply(sol,ax);sol.copy_(ax/ax.norm())
    apply(sol,ax);lam=(sol*ax).sum().item()
    lr=.8/lam
    report['sgd'].update(lr=lr,lambda_max_estimate=lam,
        calibration_s=time.perf_counter()-start,optimizer='torch.optim.SGD, momentum=0, weight_decay=0',
        gradient='approximate A_fp4(x)-b; no backward/STE through quantization',
        precision='native FP4 products, FP32 accumulation/state; FP64 true residual checks',
        history=[],targets={str(t):None for t in report['targets']})
    del sol;gc.collect();torch.cuda.empty_cache()
    # Release unused CG allocations before constructing a separate Torch cache.
    lib.flushPoolQuda(0)  # QUDA_MEMORY_DEVICE
    lib.flushPoolQuda(3)  # QUDA_MEMORY_HOST_PINNED
    start=time.perf_counter()
    gauge=load_qdp_gauge(a.gauge,(56,56,56,96),dtype=torch.float32,device='cuda')
    net=CachedWilsonNetwork(gauge,(56,56,56,96),fp4=True,fused=True)
    forward=net.compiled_forward()
    rhs=bsite.float();del bsite
    x=torch.zeros_like(rhs,requires_grad=True)
    optimizer=torch.optim.SGD([x],lr=lr,momentum=0,weight_decay=0)
    with torch.no_grad():forward(x);sync()
    # Cache construction/JIT leaves large, unused allocator blocks. Return them
    # before the long loop; retain only live tensors and compiled kernels.
    gc.collect();torch.cuda.empty_cache()
    report['sgd']['setup_compile_warmup_s']=time.perf_counter()-start
    save();print('SGD READY '+json.dumps({k:v for k,v in report['sgd'].items() if k not in ('history','targets')}),flush=True)
    start=time.perf_counter();update_total=check_total=0.
    reason='iteration_limit'
    with torch.no_grad():
        for step in range(1,a.steps+1):
            t=time.perf_counter()
            x.grad=forward(x).sub_(rhs)
            optimizer.step();optimizer.zero_grad(set_to_none=True);sync()
            update_s=time.perf_counter()-t;update_total+=update_s
            t=time.perf_counter()
            native=to_quda_native(x.double())
            true=residual_native(native);del native
            check_s=time.perf_counter()-t;check_total+=check_s
            elapsed=time.perf_counter()-start
            row={'step':step,'update_s':update_s,'check_s':check_s,
                 'update_total_s':update_total,'validated_wall_s':elapsed,'true_residual':true}
            report['sgd']['history'].append(row)
            for target in report['targets']:
                if true<=target and report['sgd']['targets'][str(target)] is None:
                    report['sgd']['targets'][str(target)]=dict(row)
            report['sgd'].update(iterations=step,final_true_residual=true,
                                update_total_s=update_total,check_total_s=check_total,validated_wall_s=elapsed)
            save()
            if step==1 or step%10==0:print('SGD '+json.dumps(row),flush=True)
            if not torch.isfinite(torch.tensor(true)) or true>10:
                reason='diverged';break
            if true<=min(report['targets']):reason='all_targets_reached';break
            # Stop only after substantial evidence of a residual floor: no 1%
            # improvement across two successive 40-step windows, after 120 steps.
            history=report['sgd']['history']
            if step>=120 and step%40==0:
                old=min(r['true_residual'] for r in history[-80:-40])
                new=min(r['true_residual'] for r in history[-40:])
                if new>=.99*old:
                    reason='plateau: less than 1% improvement over successive 40-step windows';break
    times=[r['update_s'] for r in report['sgd']['history']]
    report['sgd'].update(stop_reason=reason,mean_step_s=statistics.mean(times),median_step_s=statistics.median(times))
    report['completed']=True;save();print('DONE '+json.dumps({k:v for k,v in report['sgd'].items() if k!='history'}),flush=True)
finally:
    lib.quda_replay_finalize()
