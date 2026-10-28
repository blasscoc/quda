"""Official torch.optim.Muon + packed FP4 forward; exact QUDA diagnostics."""
import argparse,copy,ctypes,gc,json,os,statistics,time
from pathlib import Path
import torch
from packed_fp4 import PackedWilsonFP4,pack_vector,unpack_vector
from wilson_network import to_quda_native

p=argparse.ArgumentParser();p.add_argument('--bridge',required=True);p.add_argument('--gauge',required=True)
p.add_argument('--output',default='muon_fp4_results.json');a=p.parse_args()
dest=Path(a.output)
if dest.exists():raise SystemExit('Saved results already exist; choose a new --output to remeasure')
assert (Path(os.environ['QUDA_RESOURCE_PATH'])/'tunecache.tsv').exists()
report={'model':'packed_forward_56x96.pt','dims':[56,56,56,96],'gauge':a.gauge,'kappa':.156,'seed':45,
        'optimizer':'torch.optim.Muon','matrix_shape':[56**3*96//2,24],
        'matrix_layout':'checkerboard sites by spin/color/real-imag components',
        'weight_decay':0.,'momentum':.95,'nesterov':True,'ns_steps':5,'adjust_lr_fn':'original',
        'forward_precision':'packed FP4 in/out, FP32 register accumulation',
        'optimizer_precision':'FP32 master parameter/momentum; official Muon BF16 Newton-Schulz',
        'gradient':'unpack(A_fp4(pack(x))) - b; explicitly supplied approximate gradient, no STE/backward',
        'primary_residual':'||A_FP64*unpack(pack(x))-b|| / ||b||',
        'targets':[.1,.01,.001],'pilots':[],'selected':None,'completed':False}
def save():
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(dest)
def sync():torch.cuda.synchronize()
def ptr(t):return ctypes.c_void_p(t.data_ptr())
lib=ctypes.CDLL(a.bridge);I=ctypes.c_int;D=ctypes.c_double;P=ctypes.c_void_p
lib.quda_replay_init.argtypes=[I]*4+[ctypes.c_char_p,D,D,ctypes.c_ulonglong];lib.quda_replay_init.restype=I
lib.quda_replay_apply_mdagm_d2d.argtypes=[P,P];lib.quda_replay_apply_mdagm_d2d.restype=None
lib.quda_replay_finalize.restype=None
lib.flushPoolQuda.argtypes=[I];lib.flushPoolQuda.restype=None
t=time.perf_counter();rc=lib.quda_replay_init(56,56,56,96,a.gauge.encode(),5.6,.156,20461026)
if rc:raise RuntimeError(rc)
report['quda_setup_s']=time.perf_counter()-t
try:
 torch.manual_seed(45);torch.backends.cuda.matmul.allow_tf32=False
 net=PackedWilsonFP4.load(report['model']);v=net.v
 bsite=torch.randn(1,v,4,3,2,device='cuda',dtype=torch.float64)
 b=to_quda_native(bsite);bnorm2=b.square().sum();rhs=bsite.float();del bsite
 ax=torch.empty_like(b)
 def exact_metrics(site):
    native=to_quda_native(site.double())
    lib.quda_replay_apply_mdagm_d2d(ptr(native),ptr(ax));sync()
    quadratic=(.5*(native*ax).sum()-(native*b).sum()).item()
    ax.sub_(b);r2=(ax.square().sum()/bnorm2).item()
    return {'residual':r2**.5,'normalized_residual_loss':.5*r2,
            'quadratic_loss':quadratic,'normalized_quadratic_loss':quadratic/bnorm2.item()}
 forward=torch.compile(net,fullgraph=True)
 x=torch.zeros(v,24,device='cuda',dtype=torch.float32)
 with torch.no_grad():
    q,s=pack_vector(x.reshape_as(rhs));forward(q,s);sync();exact_metrics(x.reshape_as(rhs))
 del q,s
 lib.flushPoolQuda(0);lib.flushPoolQuda(3);gc.collect();torch.cuda.empty_cache()
 best=None;best_state=None
 def make_optimizer(lr):
    return torch.optim.Muon([x],lr=lr,weight_decay=0.,momentum=.95,nesterov=True,ns_steps=5,adjust_lr_fn='original')
 def warm_optimizer(opt):
    t=time.perf_counter();x.zero_();x.grad=torch.zeros_like(x)
    step=torch.compile(opt.step)
    step();sync();opt.zero_grad(set_to_none=True)
    # Zero update leaves the zero initial guess and momentum unchanged.
    assert torch.count_nonzero(x).item()==0
    return step,time.perf_counter()-t
 def iteration(opt,step,number):
    t=time.perf_counter();q,s=pack_vector(x.reshape_as(rhs));sync();pack_s=time.perf_counter()-t
    t=time.perf_counter();out=forward(q,s);sync();forward_s=time.perf_counter()-t
    t=time.perf_counter();x.grad=unpack_vector(*out).sub_(rhs).reshape_as(x);sync();gradient_s=time.perf_counter()-t
    t=time.perf_counter();step();opt.zero_grad(set_to_none=True);sync();optimizer_s=time.perf_counter()-t
    t=time.perf_counter();pq,ps=pack_vector(x.reshape_as(rhs));actual=unpack_vector(pq,ps)
    metrics=exact_metrics(actual)
    master=exact_metrics(x.reshape_as(rhs)) if number%10==0 else None
    sync();check_s=time.perf_counter()-t
    row={'step':number,'pack_s':pack_s,'forward_s':forward_s,'gradient_s':gradient_s,'optimizer_s':optimizer_s,
         'update_s':pack_s+forward_s+gradient_s+optimizer_s,'check_s':check_s,
         'true_residual':metrics['residual'],'metrics':metrics,'master_metrics':master}
    return row
 with torch.no_grad():
    for lr in (.25,.75,2.0):
        opt=make_optimizer(lr);step,warm=warm_optimizer(opt)
        trial={'lr':lr,'warmup_compile_s':warm,'history':[]};report['pilots'].append(trial)
        print('PILOT '+json.dumps({'lr':lr,'warmup_compile_s':warm}),flush=True)
        for n in range(1,21):
            row=iteration(opt,step,n);trial['history'].append(row);save()
            if n==1 or n%5==0:print('MUON '+json.dumps(dict(lr=lr,**row)),flush=True)
            if not torch.isfinite(torch.tensor(row['true_residual'])) or row['true_residual']>10:break
        if best is None or trial['history'][-1]['true_residual']<best['history'][-1]['true_residual']:
            best=trial;best_state=(x.clone(),copy.deepcopy(opt.state_dict()))
        del opt,step;gc.collect();torch.cuda.empty_cache()
    selected=copy.deepcopy(best);report['selected']=selected
    opt=make_optimizer(best['lr']);step,warm=warm_optimizer(opt)
    x.copy_(best_state[0]);opt.load_state_dict(best_state[1]);del best_state
    print('SELECTED lr='+str(best['lr']),flush=True)
    selected['stop_reason']='100-step limit'
    for n in range(len(selected['history'])+1,101):
        row=iteration(opt,step,n);selected['history'].append(row);save()
        if n%10==0:print('SELECTED '+json.dumps(row),flush=True)
        if row['true_residual']<=.001:selected['stop_reason']='all targets reached';break
        if not torch.isfinite(torch.tensor(row['true_residual'])) or row['true_residual']>10:
            selected['stop_reason']='divergence';break
        if n>=60 and n%20==0:
            h=selected['history'];old=min(r['true_residual'] for r in h[-40:-20]);new=min(r['true_residual'] for r in h[-20:])
            if new>=.99*old:selected['stop_reason']='less than 1% improvement across two 20-step windows';break
    for trial in report['pilots']+[selected]:
        h=trial['history'];trial['best_residual']=min(r['true_residual'] for r in h)
        trial['final_residual']=h[-1]['true_residual'];trial['iterations']=len(h)
        trial['mean_times']={k:statistics.mean(r[k] for r in h) for k in ('pack_s','forward_s','gradient_s','optimizer_s','update_s','check_s')}
        trial['targets']={}
        for tol in report['targets']:
            hit=next((i for i,r in enumerate(h) if r['true_residual']<=tol),None)
            trial['targets'][str(tol)]=None if hit is None else {'iterations':hit+1,
                'update_s':sum(r['update_s'] for r in h[:hit+1]),'validated_s':sum(r['update_s']+r['check_s'] for r in h[:hit+1])}
    report['completed']=True;save()
    print('DONE '+json.dumps({k:v for k,v in selected.items() if k!='history'}),flush=True)
finally:
 lib.quda_replay_finalize()
