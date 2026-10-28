"""Run a complete HMC experiment from a versioned config.json."""
import argparse,json,time
from pathlib import Path
from hmc_config import HMCConfig,revision

def build_model(cfg):
    from compact_hmc import HMC
    d=cfg.data;p=d['physics'];q=d['precision']
    kw=dict(kappa=p['kappa'],beta=p['beta'],tile=d['execution']['tile_sites'],boundary=p['temporal_boundary'])
    if q['format']=='fp4':
        from fp4_roundoff import ResearchHMC
        center=q['gauge_representation']=='identity_plus_correction';dual=q['solver_components']==2
        mode='center_dual_mom' if q['momentum_components']==2 else ('center_dual' if center and dual else 'center' if center else 'dual' if dual else 'baseline')
        h=ResearchHMC(p['lattice'],mode,q['solver_gain'],**kw)
    else:h=HMC(p['lattice'],q['format'],**kw)
    h.diagnostics=d['diagnostics']['enabled'];h.residual_every=d['diagnostics']['residual_every']
    h.hamiltonian_each_stage=d['diagnostics']['hamiltonian_each_stage'];h.stable_energy=q['energy_reduction']=='fp64'
    h.solver_refresh_intervals={k:v['residual_refresh_every'] for k,v in d['solvers'].items()}
    return h

def clean(v):
    import math
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,list):return [clean(x) for x in v]
    return None if isinstance(v,float) and not math.isfinite(v) else v

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--set',action='append',default=[],metavar='PATH=JSON')
    p.add_argument('--run-dir',type=Path,default=None);p.add_argument('--resume',type=Path);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    try:
        raw=HMCConfig.from_pretrained(a.config).to_dict()
        for item in a.set:
            key,value=item.split('=',1);parts=key.split('.');node=raw
            for part in parts[:-1]:node=node[part]
            if parts[-1] not in node:raise ValueError('Unknown override: '+key)
            node[parts[-1]]=json.loads(value)
        cfg=HMCConfig.from_dict(raw)
    except (ValueError,KeyError,TypeError,OSError) as e:p.error(str(e))
    if a.validate_only:print(json.dumps(cfg.data,indent=2));return
    import torch
    from hmc_layers import HMCTrajectory
    d=cfg.data;run=a.run_dir or Path('runs')/time.strftime('%Y%m%d-%H%M%S')
    if (run/'checkpoint.pt').exists() and not a.resume:p.error('Run already contains a checkpoint; use --resume or a new --run-dir')
    saved=torch.load(a.resume,map_location='cpu',weights_only=True) if a.resume else None
    if saved:
        if saved.get('format')!='configured-hmc-v1':p.error('Expected a configured-hmc-v1 checkpoint')
        if HMCConfig.from_dict(saved['config']).resume_contract()!=cfg.resume_contract():p.error('Resume configuration mismatch')
    torch.manual_seed(d['seed']);torch.backends.cuda.matmul.allow_tf32=d['precision']['format']=='tf32'
    writer_class=None
    if d['logging']['tensorboard']:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer_class=SummaryWriter
        except ImportError as e:p.error('TensorBoard logging requested but tensorboard is not installed: '+str(e))
    h=build_model(cfg);completed=0;solver_step=0
    if saved:
        h.gauge.q.copy_(saved['gauge_q']);h.gauge.s.copy_(saved['gauge_s']);h.refresh()
        torch.cuda.set_rng_state(saved['cuda_rng']);torch.set_rng_state(saved['cpu_rng']);completed=saved['completed'];solver_step=saved.get('solver_step',0)
        del saved
    else:h.cold()
    model=HMCTrajectory(h);run.mkdir(parents=True,exist_ok=True);cfg.save_pretrained(run)
    code=revision();(run/'provenance.json').write_text(json.dumps(dict(code=code,resume=str(a.resume) if a.resume else None,torch_version=str(torch.__version__),architecture=str(model)),indent=2)+'\n')
    writer=writer_class(str(run/'tensorboard')) if writer_class else None
    if writer:writer.add_text('experiment/config',json.dumps(d,indent=2),completed)
    print(model,flush=True)
    for n in range(d['execution']['trajectories']):
        h.timings={};h.solves=[];h.breakdowns.zero_();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        result=model(cfg)
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        result.update(trajectory=completed+n+1,seconds=seconds,config=cfg.to_dict(),code=code,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      timings=h.timings,solves=h.solves,solver_breakdowns=int(h.breakdowns),experimental=True)
        if writer:
            for solve in h.solves:
                for row in solve['iterations']:
                    solver_step+=1
                    for key in ('seconds','operator_seconds','diagnostic_seconds','recursive_relative_residual','recomputed_relative_residual'):
                        value=row.get(key)
                        if value is not None:writer.add_scalar('solver/'+solve['label']+'/'+key,value,solver_step)
            for key in ('h_initial','h_final','delta_h','seconds','peak_allocated_bytes'):
                writer.add_scalar('trajectory/'+key,result[key],completed+n+1)
            writer.add_scalar('trajectory/accepted',int(result['accepted']),completed+n+1)
            for stage in result['hamiltonian_transitions']:
                for side in ('before','after'):
                    for key,value in stage[side].items():writer.add_scalar('energy/'+stage['stage']+'/'+side+'/'+key,value,completed+n+1)
            writer.flush()
        start=time.perf_counter();tmp=run/'checkpoint.pt.tmp' 
        torch.save(dict(format='configured-hmc-v1',config=cfg.to_dict(),code=code,completed=completed+n+1,
                        solver_step=solver_step,gauge_q=h.gauge.q,gauge_s=h.gauge.s,cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state()),tmp)
        tmp.replace(run/'checkpoint.pt');result['checkpoint_seconds']=time.perf_counter()-start
        line=json.dumps(clean(result),allow_nan=False)
        with (run/'results.jsonl').open('a') as f:f.write(line+'\n')
        print(line,flush=True)
    if writer:writer.close()

if __name__=='__main__':main()
