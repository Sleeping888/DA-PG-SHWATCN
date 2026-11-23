"""Two concurrent workers; unchanged training with verified read-only dataset cache.

The only training-function adapter streams prediction CSV export in chunks.
STV uses the original dataset builder to obtain its own normalization context.
"""
import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone, timedelta


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def save(p, obj):
    p = Path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def load_module(root):
    spec = importlib.util.spec_from_file_location('comparison_source', root/'source_snapshot.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify(root):
    plan = read(root/'plan.json')
    assert sha(root/'source_snapshot.py') == plan['source_sha256']
    assert sha(root/'runner.py') == plan['runner_sha256']
    assert sha(Path(plan['dataset_cache'])/'ready.json') == plan['cache_ready_sha256']
    for data in plan['data_files']:
        stat = Path(data['path']).stat()
        assert stat.st_size == data['size'] and stat.st_mtime_ns == data['mtime_ns'], 'Raw data changed'
    return plan


def stream_predictions(pred, true, times, turbines, cfg, mode):
    import pandas as pd
    path = Path(cfg.out_dir)/f'test_predictions_{mode}.csv'
    for start in range(0, len(true), 64):
        rows = []
        for ti in range(start, min(start+64, len(true))):
            for hh in range(true.shape[-1]):
                stamp = pd.Timestamp(times[ti, hh])
                for j, turbine in enumerate(turbines):
                    rows.append(dict(target_time=stamp, horizon=hh+1, turbine=turbine,
                                     true_kw=float(true[ti,j,hh]), pred_kw=float(pred[ti,j,hh])))
        pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig',
                                 mode='w' if start == 0 else 'a', header=start == 0)


def adapt_export(module):
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.train_one_experiment)))
    f = tree.body[0]
    start = next(i for i,x in enumerate(f.body) if isinstance(x, ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id == 'rows' for t in x.targets))
    stop = next(i for i,x in enumerate(f.body[start:], start) if isinstance(x, ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == 'result' for t in x.targets))
    # Replace only rows construction + CSV export. The optimization, validation,
    # target scaling and reported metrics remain the original AST nodes.
    f.body[start:stop] = ast.parse('stream_predictions(pred_matrix_kw, true_matrix_kw, test["times"], turbines, cfg, mode)').body
    env = dict(vars(module), stream_predictions=stream_predictions)
    exec(compile(ast.fix_missing_locations(tree), '<streaming_export_adapter>', 'exec'), env)
    module.train_one_experiment = env['train_one_experiment']


def worker(root, seed, mode):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    plan = verify(root)
    assert seed in plan['seeds'] and mode in plan['models']
    m = load_module(root)
    original_build = m.build_dataset
    cache = Path(plan['dataset_cache'])
    meta = read(cache/'ready.json')
    holder = {}
    def cached_build(cfg, requested):
        if requested == 'stv_pgnn_adapted':
            result = original_build(cfg, requested)
        else:
            ignored = {'seed','experiments','out_dir','device'}
            for k,v in meta['config'].items():
                if k not in ignored:
                    assert getattr(cfg,k) == v, f'Cache config mismatch: {k}'
            physical = requested in {'full','full_no_adapt','no_hetero_adapt','no_hetero','stgcn'}
            datasets = []
            for split in ['train','val','test']:
                d = {}
                for key in ['X_time','Y_time','nbr_idx','edge_attr','edge_mask','role','Y_matrix','times']:
                    arr = np.load(cache/f'{split}_{key}.npy', mmap_mode='c', allow_pickle=False)
                    if not physical and key in {'nbr_idx','edge_attr','edge_mask','role'}:
                        arr = np.zeros(arr.shape, dtype=arr.dtype)
                    d[key] = arr
                datasets.append(d)
            result = (*datasets, meta['turbines'], m.pd.DatetimeIndex(np.load(cache/'pred_times.npy')))
            print(f'[Cache] verified original preprocessing; {meta["samples"]}', flush=True)
        holder['val'], holder['test'] = result[1:3]
        return result
    m.build_dataset = cached_build
    original_evaluate = m.evaluate
    tracking = dict(calls=0, best=float('inf'), epoch=0)
    folder = root/f'seed_{seed}'/mode
    def evaluate(model, data, cfg, n):
        result = original_evaluate(model,data,cfg,n)
        if data is holder.get('val'):
            tracking['calls'] += 1
            score = result[0]['node_avg_rmse']
            if score < tracking['best']-1e-6:
                tracking.update(best=score, epoch=tracking['calls'])
            save(folder/'progress.json', dict(seed=seed, model=mode, validation_call=tracking['calls'],
                  val_node_avg_rmse=score, best_epoch=tracking['epoch'], updated_at=now()))
        if data is holder.get('test'):
            torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            seed=seed, model=mode, best_epoch=tracking['epoch'],
                            config=m.asdict(cfg)), folder/'checkpoint_best.pt')
        return result
    m.evaluate = evaluate
    adapt_export(m)
    sys.argv = [str(root/'source_snapshot.py'), '--seed',str(seed),'--experiments',mode,'--out_dir',str(folder)]
    m.main()


def aggregate(root, plan):
    import pandas as pd
    rows, horizons = [], []
    for seed in plan['seeds']:
        for mode in plan['models']:
            f = root/f'seed_{seed}'/mode
            if not (f/'completed.json').exists():
                continue
            a = pd.read_csv(f/'summary_results.csv')
            assert len(a) == 1 and a.iloc[0]['case'] == mode
            a.insert(0,'seed',seed)
            rows.append(a)
            b = pd.read_csv(f/'summary_results_by_horizon.csv')
            assert set(b.horizon) == {'avg', *[f't+{h}' for h in range(1,7)]}
            b.insert(0,'seed',seed)
            horizons.append(b)
    if not rows:
        return
    wide = pd.concat(rows,ignore_index=True)
    wide.to_csv(root/'all_metrics_per_seed.csv',index=False,encoding='utf-8-sig')
    long = pd.concat(horizons,ignore_index=True)
    long.to_csv(root/'all_horizons_per_seed.csv',index=False,encoding='utf-8-sig')
    metrics = [k for k in long if k.startswith(('node_', 'farm_'))]
    summary = long.groupby(['case','model','horizon'],sort=False)[metrics].agg(['mean','std','count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(root/'mean_std_by_horizon.csv',index=False,encoding='utf-8-sig')
    if len(wide) == len(plan['seeds'])*len(plan['models']):
        n = len(plan['seeds'])
        seed_text = '、'.join(map(str,plan['seeds']))
        lines = ['# 模型对比结果', '', f'种子{seed_text}，全部{len(wide)}组均已完成；未筛选种子。',
                 '以下为h1–h6算术AVG指标，单机和场级RMSE均以kW计。多个种子时报告均值±样本标准差；单个种子时仅报告该种子结果。', '',
                 '| 模型 | AVG单机RMSE | AVG单机MAE | AVG场级RMSE |', '|---|---:|---:|---:|']
        for mode in plan['models']:
            g = wide[wide.case.eq(mode)]
            assert len(g) == n
            lines.append('| '+mode+' | '+' | '.join((f'{g[k].mean():.3f}±{g[k].std():.3f}' if n>1 else f'{g[k].iloc[0]:.3f}') for k in
                         ['test_node_avg_rmse','test_node_avg_mae','test_farm_avg_rmse'])+' |')
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def controller(root):
    import msvcrt
    import psutil
    plan = verify(root)
    pending = [(seed,model) for model in plan['models'] for seed in plan['seeds']]
    active = {}
    total = len(pending)
    concurrency = int(plan.get('concurrency',2))
    state = dict(status='running', controller_pid=os.getpid(), total=total, completed=0,
                 concurrency=concurrency, started_at=now(), active=[])
    with (root/'controller.lock').open('a+b') as lock:
        lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            while pending or active:
                if (root/'STOP_REQUESTED').exists():
                    state['status'] = 'stopping_after_active_cases'
                    pending.clear()
                for pid, (p, seed, model, log, started) in list(active.items()):
                    if p.poll() is None:
                        continue
                    log.close()
                    del active[pid]
                    folder = root/f'seed_{seed}'/model
                    if p.returncode:
                        state.update(status='failed', error=f'{seed}/{model}: exit {p.returncode}')
                        # Retain other active training, but do not launch more.
                        pending.clear()
                    else:
                        assert (folder/'checkpoint_best.pt').exists()
                        save(folder/'completed.json',dict(seed=seed,model=model,exit_code=0,
                             elapsed_seconds=time.time()-started,finished_at=now(),
                             summary_sha256=sha(folder/'summary_results.csv')))
                        state['completed'] += 1
                        aggregate(root,plan)
                while pending and len(active) < concurrency and state['status'] == 'running':
                    # Wait for memory/disk headroom without changing batch/model settings.
                    if psutil.virtual_memory().available < 3*1024**3:
                        break
                    if psutil.disk_usage(str(root)).free < 2*1024**3:
                        raise RuntimeError('Less than 2 GiB free disk; no more training launched')
                    seed, model = pending.pop(0)
                    folder = root/f'seed_{seed}'/model
                    folder.mkdir(parents=True,exist_ok=False)
                    log = (folder/'console.log').open('ab',buffering=0)
                    env = dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1',PYTHONIOENCODING='utf-8')
                    p = subprocess.Popen([sys.executable,'-u',str(root/'runner.py'),'--root',str(root),
                         '--seed',str(seed),'--model',model],cwd=root,env=env,stdout=log,
                         stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)
                    active[p.pid] = (p,seed,model,log,time.time())
                    print(f'Started {seed}/{model}, pid={p.pid}',flush=True)
                state.update(updated_at=now(), pending=len(pending),
                     active=[dict(pid=pid,seed=x[1],model=x[2]) for pid,x in active.items()])
                save(root/'status.json',state)
                if pending or active:
                    time.sleep(5)
            if state['status'] == 'running':
                state['status'] = 'complete'
            elif state['status'] == 'stopping_after_active_cases':
                state['status'] = 'stopped'
            state['finished_at'] = now()
            save(root/'status.json',state)
        except Exception as exc:
            state.update(status='failed',error=repr(exc),updated_at=now())
            save(root/'status.json',state)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int)
    p.add_argument('--model')
    a = p.parse_args()
    if a.model:
        worker(a.root.resolve(),a.seed,a.model)
    else:
        controller(a.root.resolve())
