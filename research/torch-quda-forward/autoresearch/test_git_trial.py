import json,subprocess,sys,tempfile
from pathlib import Path
tool=Path(__file__).with_name('git_trial.py')
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)/'repo';root.mkdir();work=Path(tmp)/'trial'
    def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True,stderr=subprocess.DEVNULL).strip()
    git('init');git('config','user.name','Test');git('config','user.email','test@example.invalid')
    (root/'candidate.txt').write_text('baseline\n');git('add','candidate.txt');git('commit','-m','baseline')
    original=git('rev-parse','HEAD')
    def run(*args,ok=True):
        p=subprocess.run([sys.executable,str(tool),'--repo',str(root),'--campaign','test',*map(str,args)],text=True,capture_output=True)
        assert (p.returncode==0)==ok,p.stdout+p.stderr
        return p.stdout.strip()
    run('start','one','--directory',work)
    (work/'candidate.txt').write_text('candidate\n')
    run('save','--worktree',work,'--message','experiment','candidate.txt')
    commit=subprocess.check_output(['git','-C',str(work),'rev-parse','HEAD'],text=True).strip()
    receipt=Path(tmp)/'receipt.json'
    row=dict(commit=commit,status='keep',contract_id='fixed',evidence='raw.json',seconds=1.,repeat_count=3,
             validation_passed=True,solver_quality_passed=True,hamiltonian_passed=True,memory_scaling_passed=True)
    receipt.write_text(json.dumps({**row,'memory_scaling_passed':False}));run('record',receipt,ok=False)
    receipt.write_text(json.dumps(row));run('record',receipt)
    assert run('best')==commit
    receipt.write_text(json.dumps({**row,'status':'discard'}));run('record',receipt,ok=False)
    assert run('best')==commit and git('rev-parse','HEAD')==original
    assert (root/'candidate.txt').read_text()=='baseline\n'
    run('save','--worktree',work,'--message','bad','../outside',ok=False)
print('PASS git trials: isolated branch, explicit save, gate refusal, immutable evidence, preserved source and best ref')
