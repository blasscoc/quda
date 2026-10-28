"""GPU QUDA CG baseline on the same gauge/RHS as benchmark_network.py."""
import argparse, ctypes, hashlib, json, os, time
from pathlib import Path
import torch
from wilson_network import to_quda_native

p=argparse.ArgumentParser()
p.add_argument('--bridge',required=True)
p.add_argument('--gauge',required=True)
p.add_argument('--output',default='quda_cg_benchmark.json')
p.add_argument('--refresh',action='store_true',help='Measure again even when an identical completed result exists')
p.add_argument('--record-existing',help='Explicitly register a trusted completed result from this same build and input; does not benchmark')
a=p.parse_args()
root=Path(__file__).resolve().parent
cache=Path(os.environ.setdefault('QUDA_RESOURCE_PATH',str(root/'quda-cache')))
cache.mkdir(parents=True,exist_ok=True)
results=root/'benchmark-results';results.mkdir(exist_ok=True)
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
library=next((Path(d)/'libquda.so' for d in os.environ.get('LD_LIBRARY_PATH','').split(':')
              if d and (Path(d)/'libquda.so').is_file()),None)
if library is None:raise RuntimeError('Set LD_LIBRARY_PATH to the exact QUDA build; use run_cg_benchmark.sh')
identity={'schema':1,'gauge_sha256':digest(a.gauge),'bridge_sha256':digest(a.bridge),
          'quda_sha256':digest(library),'script_sha256':digest(__file__),
          'layout_sha256':digest(root/'wilson_network.py'),'torch':torch.__version__,
          'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
          'dims':[56,56,56,96],'seed':45,'kappa':.156,'tol':1e-8,'maxiter':1000,
          'container':'nvcr.io/nvidia/pytorch:26.09-py3'}
key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
saved=results/(key+'.json')
if a.record_existing:
    previous=json.loads(Path(a.record_existing).read_text())
    expected={'dims':[56,56,56,96],'gauge':a.gauge,'seed':45,'kappa':.156,
              'tolerance':1e-8,'maxiter':1000,'torch':torch.__version__}
    if any(previous.get(k)!=v for k,v in expected.items()) or not previous.get('converged'):
        raise ValueError('Existing result does not match the requested completed benchmark')
    previous.update(identity=identity,cache_key=key,completed=True,
                    provenance='Explicitly imported from pre-cache runner; measured previously, not during import',
                    imported_from=a.record_existing)
    saved.write_text(json.dumps(previous,indent=2))
    print('Registered existing measurement: '+str(saved),flush=True)
    raise SystemExit(0)
if saved.exists() and not a.refresh:
    previous=json.loads(saved.read_text())
    if previous.get('completed'):
        Path(a.output).write_text(json.dumps(previous,indent=2))
        print('REUSED SAVED RESULT (no new GPU benchmark): '+str(saved),flush=True)
        print(json.dumps(previous),flush=True)
        raise SystemExit(0)
run_started=time.time()
lib=ctypes.CDLL(a.bridge)
ptr=ctypes.c_void_p; dbl=ctypes.c_double; integer=ctypes.c_int
lib.quda_replay_init.argtypes=[integer]*4+[ctypes.c_char_p,dbl,dbl,ctypes.c_ulonglong]
lib.quda_replay_init.restype=integer
lib.quda_replay_apply_mdagm_d2d.argtypes=[ptr,ptr]
lib.quda_replay_apply_mdagm_d2d.restype=None
lib.quda_replay_solve_cg_d2d.argtypes=[ptr,ptr,dbl,integer,ctypes.POINTER(integer)]+[ctypes.POINTER(dbl)]*3
lib.quda_replay_solve_cg_d2d.restype=None
lib.quda_replay_finalize.restype=None
dims=(56,56,56,96);v=56**3*96//2
start=time.perf_counter()
rc=lib.quda_replay_init(*dims,a.gauge.encode(),5.6,.156,20461026)
if rc:raise RuntimeError(f'QUDA init {rc}')
report={'dims':dims,'gauge':a.gauge,'seed':45,'kappa':.156,'boundary':'periodic',
        'precision':'QUDA FP64 including sloppy','init_s':time.perf_counter()-start,
        'rhs':'same seeded Gaussian as full_56x96_benchmark.json; UKQCD native vector_reals=4',
        'tolerance':1e-8,'maxiter':1000,'torch':torch.__version__,
        'identity':identity,'cache_key':key,'completed':False,'started_unix':run_started,
        'tuning_cache':str(cache),'tuning_cache_present_at_start':(cache/'tunecache.tsv').exists()}
def save():
    report['elapsed_wall_s']=time.time()-run_started
    for dest in (Path(a.output),saved):
        tmp=dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2));tmp.replace(dest)
    print(json.dumps(report),flush=True)
try:
    torch.manual_seed(45)
    b=to_quda_native(torch.randn(1,v,4,3,2,device='cuda',dtype=torch.float64),vector_reals=4)
    y=torch.empty_like(b);x=torch.zeros_like(b)
    def apply(inp,out):
        lib.quda_replay_apply_mdagm_d2d(inp.data_ptr(),out.data_ptr())
        torch.cuda.synchronize()
    apply(b,y)
    samples=[]
    for _ in range(3):
        t=time.perf_counter();apply(b,y);samples.append(time.perf_counter()-t)
    report['operator_s_samples']=samples
    report['operator_s_median']=sorted(samples)[1]
    save()
    iters=integer();res=dbl();secs=dbl();gflops=dbl()
    t=time.perf_counter()
    lib.quda_replay_solve_cg_d2d(b.data_ptr(),x.data_ptr(),1e-8,1000,
        ctypes.byref(iters),ctypes.byref(res),ctypes.byref(secs),ctypes.byref(gflops))
    torch.cuda.synchronize()
    report.update(solve_call_wall_s=time.perf_counter()-t,solve_internal_s=secs.value,
                  iterations=iters.value,quda_reported_residual=res.value)
    apply(x,y)
    y.sub_(b)
    true=(y.norm()/b.norm()).item()
    report.update(independent_relative_residual=true,converged=bool(true<=1e-8),
                  timing_note='one solve, includes invertQuda first-use overhead; operator median is warmed; bridge CPU staging included in call wall time')
    save()
finally:
    lib.quda_replay_finalize()
report['completed']=True
save()
