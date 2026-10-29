#!/usr/bin/env python3
"""Isolated Git trials and evidence-backed best refs; never reset shared work.

start ID --base REF --directory PATH
save --worktree PATH --message TEXT FILE [FILE ...]
record RECEIPT.json   (status=baseline/keep/discard/crash; SHA, evidence and measurements)
best                  (refs/autoresearch/CAMPAIGN/best)
"""
import argparse,json,math,re,subprocess
from pathlib import Path

def git(root,*args,check=True):
    result=subprocess.run(['git','-C',str(root),*args],text=True,capture_output=True)
    if check and result.returncode:raise ValueError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()

def slug(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',value):raise argparse.ArgumentTypeError('Use letters, digits, _ and -')
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--campaign',type=slug,required=True)
    sub=parser.add_subparsers(dest='action',required=True)
    start=sub.add_parser('start');start.add_argument('id',type=slug);start.add_argument('--base',default='HEAD');start.add_argument('--directory',type=Path,required=True)
    save=sub.add_parser('save');save.add_argument('--worktree',type=Path,required=True);save.add_argument('--message',required=True);save.add_argument('files',nargs='+')
    rec=sub.add_parser('record');rec.add_argument('receipt',type=Path)
    sub.add_parser('best');args=parser.parse_args()
    try:
        root=Path(git(args.repo,'rev-parse','--show-toplevel'));ref=f'refs/autoresearch/{args.campaign}/best'
        if args.action=='start':
            commit=git(root,'rev-parse','--verify',args.base+'^{commit}')
            print(git(root,'worktree','add','-b',f'autoresearch/{args.campaign}/{args.id}',str(args.directory.resolve()),commit))
        elif args.action=='save':
            work=args.worktree.resolve()
            if not git(work,'branch','--show-current').startswith(f'autoresearch/{args.campaign}/'):raise ValueError('save requires this campaign’s isolated branch')
            if git(work,'diff','--cached','--name-only'):raise ValueError('Existing staged changes: review/commit them explicitly first')
            for name in args.files:
                if not (work/name).resolve().is_relative_to(work):raise ValueError('File escapes worktree')
            git(work,'--literal-pathspecs','add','--',*args.files)
            print(git(work,'commit','-m',args.message));print(git(work,'rev-parse','HEAD'))
        elif args.action=='best':
            print(git(root,'rev-parse','--verify',ref))
        else:
            row=json.loads(args.receipt.read_text());status=row.get('status')
            if status not in ('baseline','keep','discard','crash'):raise ValueError('Receipt status must be baseline/keep/discard/crash')
            commit=git(root,'rev-parse','--verify',str(row.get('commit',''))+'^{commit}')
            if not row.get('contract_id') or not row.get('evidence'):raise ValueError('Contract and raw evidence reference required')
            previous=git(root,'rev-parse','--verify',ref,check=False)
            if status in ('baseline','keep'):
                if status=='baseline' and previous:raise ValueError('A baseline/best already exists for this campaign')
                for key in ('validation_passed','solver_quality_passed','hamiltonian_passed','memory_scaling_passed'):
                    if row.get(key) is not True:raise ValueError(f'Missing passing gate: {key}')
                seconds=row.get('seconds');repeats=row.get('repeat_count')
                if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:raise ValueError('Positive finite measured seconds required')
                if type(repeats)!=int or repeats<3:raise ValueError('At least three measured repeats required')
                if previous:
                    prior=json.loads(git(root,'notes',f'--ref=autoresearch/{args.campaign}','show',previous))
                    if row['contract_id']!=prior['contract_id']:raise ValueError('Contract differs from current best')
                    if seconds>=prior['seconds']:raise ValueError('No measured speed improvement over current best')
            # Notes retain rejected experiments and evidence without changing trees.
            git(root,'notes',f'--ref=autoresearch/{args.campaign}','add','-F',str(args.receipt.resolve()),commit)
            if status in ('baseline','keep'):git(root,'update-ref',ref,commit,previous or '0'*40)
            print(json.dumps(dict(commit=commit,status=status,best_ref=ref if status in ('baseline','keep') else None)))
    except (ValueError,OSError,json.JSONDecodeError) as exc:parser.error(str(exc))

if __name__=='__main__':main()
