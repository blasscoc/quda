"""Combine saved convergence trials and standardized short memory replays."""
import csv,json
from pathlib import Path
rows=[];memory=[]
for precision in ('fp4','fp8','fp16','bf16'):
    r=json.loads(Path(f'muon_{precision}_results.json').read_text())
    assert r['completed']
    s=r['selected'];all_trials=r['pilots']+[s]
    rows.append([precision,s['lr'],s['iterations'],s['best_residual'],s['final_residual'],
                 min(t['best_residual'] for t in all_trials),s['mean_times']['forward_s'],
                 s['mean_times']['optimizer_s'],s['mean_times']['update_s'],
                 *(s['targets'][str(t)] is not None for t in (.1,.01,.001))])
    probe=Path(f'memory_probe_{precision}.json')
    if probe.exists():
        p=json.loads(probe.read_text());assert p['completed'];m=p['memory']
        steady=max(x['torch_allocated'] for x in m['samples'] if x['phase'].startswith('iteration'))
        memory.append([precision,sum(m['model_buffers'].values())/2**30,m['encoded_vector_bytes']/2**20,
            2*m['fp32_master_bytes']/2**30,steady/2**30,m['summary']['torch_peak_allocated']/2**30,
            m['summary']['torch_peak_reserved']/2**30,m['summary']['system_unavailable']/2**30])
with open('muon_precision_comparison.csv','w') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['precision','selected_lr','iterations','selected_best_residual','final_residual',
    'best_across_trials','forward_s','optimizer_s','whole_update_s','reached_1e-1','reached_1e-2','reached_1e-3']);w.writerows(rows)
with open('muon_memory_comparison.csv','w') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['precision','model_buffers_GiB','encoded_vector_MiB','master_plus_momentum_GiB',
    'max_sampled_torch_live_GiB','torch_peak_allocated_GiB','torch_peak_reserved_GiB','max_sampled_system_unavailable_GiB']);w.writerows(memory)
print('CONVERGENCE',json.dumps(rows));print('MEMORY',json.dumps(memory))
