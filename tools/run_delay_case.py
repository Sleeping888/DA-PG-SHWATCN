"""Run one archived delay/window condition using bundled data and a rebuilt cache.

Does not execute training unless invoked without --prepare-only.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--cache',type=Path,required=True,help='Output of rebuild_preprocessing.py')
p.add_argument('--output',type=Path,required=True,help='New working directory')
p.add_argument('--seed',type=int,choices=[84,101,1234],required=True)
p.add_argument('--case',choices=[d+'_L'+str(L) for L in [3,13] for d in ['physical','fixed10','fixed20']],required=True)
p.add_argument('--prepare-only',action='store_true')
a=p.parse_args()
assert not a.output.exists(),'Choose a new output directory'
cache=a.cache.resolve();work=a.output.resolve();assert (cache/'ready.json').exists()
native=ROOT/'code/as_run/delay_window';plan=json.loads((native/'plan.json').read_text(encoding='utf-8'))
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
work.mkdir(parents=True)
for name in ['source_snapshot.py','runner.py','runner_common.py']:shutil.copy2(native/name,work/name)
shutil.copytree(ROOT/'data/delay_preprocessing',work/'data')
plan.update(workspace=str(ROOT),dataset_cache=str(cache),cache_ready_sha256=sha(cache/'ready.json'),
            runner_sha256=sha(work/'runner.py'),helper_sha256=sha(work/'runner_common.py'),
            source_sha256=sha(work/'source_snapshot.py'),data_ready_sha256=sha(work/'data/ready.json'))
plan['config']['data_csv']=str(ROOT/'data/raw/sdwpf_2021_full.csv')
plan['config']['location_csv']=str(ROOT/'data/raw/sdwpf_turb_location_elevation.csv')
(work/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
folder=work/f'seed_{a.seed}'/a.case;folder.mkdir(parents=True)
if a.prepare_only:
    print('Prepared portable paths; no training executed:',work)
else:
    started=time.perf_counter()
    cmd=[sys.executable,'-u',str(work/'runner.py'),'--root',str(work),'--seed',str(a.seed),'--model',a.case]
    result=subprocess.run(cmd,cwd=work,check=True)
    (folder/'completed.json').write_text(json.dumps(dict(seed=a.seed,case=a.case,exit_code=result.returncode,
        elapsed_seconds=time.perf_counter()-started,summary_sha256=sha(folder/'summary_results.csv')),
        indent=2),encoding='utf-8')
