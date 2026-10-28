"""Strict, versioned JSON experiment configuration; no Transformers dependency."""
import copy,json,math,subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULTS={
 'schema_version':1,
 'physics':{'lattice':[32,32,32,32],'beta':5.6,'kappa':.156,'temporal_boundary':1},
 'integrator':{'method':'leapfrog','steps':1,'step_size':.001},
 'solvers':{name:{'iterations':n,'residual_refresh_every':0} for name,n in
            [('force',4),('initial_energy',4),('final_energy',4),('diagnostic_energy',4)]},
 'precision':{'format':'fp4','solver_components':1,'gauge_representation':'direct',
              'momentum_components':1,'energy_reduction':'fp32','solver_gain':1.},
 'diagnostics':{'enabled':True,'residual_every':1,'hamiltonian_each_stage':True},
 'execution':{'tile_sites':8192,'trajectories':1},'logging':{'tensorboard':False},'seed':45}

def merge(base,patch,path=''):
 if not isinstance(patch,dict):raise ValueError(f'{path or "config"} must be an object')
 out=copy.deepcopy(base)
 for k,v in patch.items():
  if k not in base:raise ValueError(f'Unknown configuration key: {path}{k}')
  out[k]=merge(base[k],v,path+k+'.') if isinstance(base[k],dict) else v
 return out

def integer(v,lo,name):
 if type(v)!=int or v<lo:raise ValueError(f'{name} must be an integer >= {lo}')

def finite(v,name,positive=False):
 if type(v) not in (int,float) or not math.isfinite(v) or (positive and v<=0):raise ValueError(f'Invalid {name}')

@dataclass(frozen=True)
class HMCConfig:
    data:dict
    @classmethod
    def from_dict(cls,patch):
        d=merge(DEFAULTS,patch)
        if type(d['schema_version'])!=int or d['schema_version']!=1:raise ValueError('Unsupported schema_version')
        p=d['physics'];dims=p['lattice']
        if not isinstance(dims,list) or len(dims)!=4 or any(type(n)!=int or n<=0 or n%2 for n in dims):raise ValueError('lattice needs four positive even integers')
        finite(p['beta'],'beta');finite(p['kappa'],'kappa',True)
        if type(p['temporal_boundary'])!=int or p['temporal_boundary'] not in (-1,1):raise ValueError('Invalid boundary')
        i=d['integrator']
        if i['method']!='leapfrog':raise ValueError('Only leapfrog is implemented')
        integer(i['steps'],1,'integrator.steps');finite(i['step_size'],'step_size',True)
        for name,s in d['solvers'].items():
            integer(s['iterations'],1,name+'.iterations');integer(s['residual_refresh_every'],0,name+'.residual_refresh_every')
        q=d['precision']
        if q['format'] not in ('fp4','fp8','fp16','bf16','tf32'):raise ValueError('Invalid format')
        for name in ('solver_components','momentum_components'):
            if type(q[name])!=int or q[name] not in (1,2):raise ValueError('Components must be 1 or 2')
        if q['gauge_representation'] not in ('direct','identity_plus_correction'):raise ValueError('Invalid gauge representation')
        if q['energy_reduction'] not in ('fp32','fp64'):raise ValueError('Invalid energy reduction')
        finite(q['solver_gain'],'solver_gain',True)
        expanded=q['solver_components']==2;center=q['gauge_representation']=='identity_plus_correction'
        if q['format']!='fp4' and (expanded or center or q['momentum_components']!=1 or q['solver_gain']!=1):raise ValueError('Representation options require FP4')
        if q['solver_gain']!=1 and not expanded:raise ValueError('solver_gain requires two-component vectors')
        if q['momentum_components']==2 and not (expanded and center):raise ValueError('Two-component momentum currently requires centered gauge and two-component solver')
        t=d['diagnostics']
        for k in ('enabled','hamiltonian_each_stage'):
            if type(t[k])!=bool:raise ValueError(k+' must be boolean')
        integer(t['residual_every'],0,'residual_every')
        for k in ('tile_sites','trajectories'):integer(d['execution'][k],1,k)
        if type(d['logging']['tensorboard'])!=bool:raise ValueError('logging.tensorboard must be boolean')
        integer(d['seed'],0,'seed')
        return cls(d)
    @classmethod
    def from_pretrained(cls,path):
        p=Path(path);return cls.from_dict(json.loads((p/'config.json' if p.is_dir() else p).read_text()))
    def save_pretrained(self,directory):
        p=Path(directory);p.mkdir(parents=True,exist_ok=True)
        (p/'config.json').write_text(json.dumps(self.data,indent=2,allow_nan=False)+'\n')
    def to_dict(self):return copy.deepcopy(self.data)
    def resume_contract(self):
        d=self.to_dict();d['execution'].pop('trajectories');d.pop('seed');d.pop('logging')
        return d  # RNG restored from checkpoint; count means additional trajectories.

def revision():
    root=Path(__file__).resolve().parent
    def git(*args):return subprocess.check_output(['git','-c',f'safe.directory={root}',*args],cwd=root,text=True).strip()
    import hashlib
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    try:return {'commit':git('rev-parse','HEAD'),'dirty':bool(git('status','--porcelain')),'source_sha256':sources}
    except (subprocess.CalledProcessError,FileNotFoundError):return {'commit':None,'dirty':None,'source_sha256':sources}
