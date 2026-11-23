"""Matched 3-delay x 2-history controls for the current original-source cohort."""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset,DataLoader

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import runner_common as common

ROOT=Path(common.read(HERE/'plan.json')['workspace']) if (HERE/'plan.json').exists() else HERE.parent
OLD=ROOT/'对比实验_1234_21_42_84_20261119_164746'
SEEDS=[84,101,1234]
CASES=[dict(name=f'{delay}_L{L}',delay=delay,L=L) for L in [3,13] for delay in ['physical','fixed10','fixed20']]


def verify(root):
    plan=common.read(root/'plan.json')
    assert common.sha(root/'runner.py')==plan['runner_sha256']
    assert common.sha(root/'runner_common.py')==plan['helper_sha256']
    assert common.sha(root/'source_snapshot.py')==plan['source_sha256']
    assert common.sha(root/'data/ready.json')==plan['data_ready_sha256']
    assert common.sha(Path(plan['dataset_cache'])/'ready.json')==plan['cache_ready_sha256']
    return plan


class Windows(Dataset):
    def __init__(self,root,split,L):
        self.L=L;plan=common.read(root/'plan.json');cache=Path(plan['dataset_cache'])
        meta=common.read(cache/'ready.json');offset={'train':0,'val':meta['samples']['train'],
                    'test':meta['samples']['train']+meta['samples']['val']}[split]
        self.origins=np.arange(2+offset,2+offset+meta['samples'][split])
        self.keep=np.flatnonzero(self.origins>=12)
        self.features=np.load(root/'data/features.npy',mmap_mode='r')
        self.arr={k:np.load(cache/f'{split}_{k}.npy',mmap_mode='r') for k in
                  ['Y_time','nbr_idx','edge_attr','edge_mask','role']}
    def __len__(self):return len(self.keep)
    def __getitem__(self,i):
        s=int(self.keep[i]);t=int(self.origins[s]);g=self.arr
        return (self.features[t-self.L+1:t+1].copy(),g['Y_time'][s].copy(),g['nbr_idx'][s].copy(),
                g['edge_attr'][s].astype(np.float32),g['edge_mask'][s].copy(),g['role'][s].copy())


def adjusted(edge,strategy):
    if strategy=='physical':return edge
    # All delay-dependent paths receive the same fixed lag, including edge bias,
    # delay penalty and historical retrieval; weights/topology/roles stay fixed.
    lag=.5 if strategy=='fixed10' else 1.
    return torch.cat([edge[...,:6],torch.where(edge[...,3:4]>0,lag,0.)],dim=-1)


@torch.no_grad()
def predict(model,loader,device,strategy):
    model.eval();out=[]
    for batch in loader:
        x,y,n,e,m,r=[a.to(device) for a in batch]
        out.append(model(x,n,adjusted(e,strategy),m,r).cpu().numpy())
    return np.clip(np.concatenate(out),0,1.5)*1500


def metrics(y,p):
    mask=np.isfinite(y)&(y>=0);result={}
    for level,yy,pp,valid,cap in [('node',y,p,mask,1500),
        ('farm',y.sum(1),p.sum(1),mask.all(1),201000)]:
        yy=yy[valid].astype(np.float64);pp=pp[valid].astype(np.float64)
        d=yy-pp;rmse=np.sqrt(np.mean(d*d));mae=np.abs(d).mean();sst=np.sum((yy-yy.mean())**2)
        result.update({f'{level}_rmse':float(rmse),f'{level}_mae':float(mae),
                       f'{level}_nrmse':float(100*rmse/cap),f'{level}_nmae':float(100*mae/cap),
                       f'{level}_r2':float(1-np.sum(d*d)/sst),f'{level}_count':len(yy)})
    return result


def worker(root,seed,name):
    torch.set_num_threads(4);plan=verify(root);case=next(x for x in CASES if x['name']==name)
    assert seed in SEEDS;source=common.load_module(root)
    cfg=source.Config(**plan['config']);cfg.seed=seed;cfg.temporal_window=case['L']
    source.set_seed(seed);folder=root/f'seed_{seed}'/name
    ds={s:Windows(root,s,case['L']) for s in ['train','val','test']}
    loaders={s:DataLoader(d,batch_size=30,shuffle=s=='train',num_workers=0,
                          generator=torch.Generator().manual_seed(seed)) for s,d in ds.items()}
    model=source.make_model(cfg,'full',6,n_turbines=134).to(cfg.device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    common.save(folder/'config.json',dict(case=case,config=asdict(cfg),parameters=sum(p.numel() for p in model.parameters()),
                   samples={s:len(d) for s,d in ds.items()},test_policy='raw finite nonnegative observed labels only'))
    val_y=np.asarray(ds['val'].arr['Y_time'])[ds['val'].keep]*1500
    best=float('inf');bad=0;history=[];started=time.perf_counter()
    for epoch in range(1,51):
        model.train();losses=[]
        for batch in loaders['train']:
            x,y,n,e,m,r=[a.to(cfg.device) for a in batch];opt.zero_grad(set_to_none=True)
            pred=model(x,n,adjusted(e,case['delay']),m,r)
            loss=(pred-y).square().mean()+cfg.farm_loss_weight*(pred.mean(1)-y.mean(1)).square().mean()
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step();losses.append(float(loss.detach()))
        vp=predict(model,loaders['val'],cfg.device,case['delay'])
        score=float(np.sqrt(np.mean((vp.astype(np.float64)-val_y.astype(np.float64))**2)))
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),val_node_pooled_rmse=score))
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        print(f'{name}/{seed} epoch {epoch:03d} val_RMSE={score:.4f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        common.save(folder/'progress.json',dict(epoch=epoch,val_node_pooled_rmse=score,updated_at=common.now()))
        if score<best-1e-6:
            best=score;bad=0
            torch.save(dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                           seed=seed,case=case,config=asdict(cfg),best_epoch=epoch),folder/'checkpoint_best.pt')
        else:
            bad+=1
            if bad>=8:break
    ck=torch.load(folder/'checkpoint_best.pt',map_location=cfg.device,weights_only=False)
    model.load_state_dict(ck['model_state_dict']);pred=predict(model,loaders['test'],cfg.device,case['delay'])
    observed=np.load(root/'data/test_observed_kw.npy')
    pooled=metrics(observed,pred)
    steps=[dict(horizon=h+1,**metrics(observed[...,h:h+1],pred[...,h:h+1])) for h in range(6)]
    np.savez_compressed(folder/'predictions.npz',pred_kw=pred)
    common.save(folder/'metrics.json',dict(pooled=pooled,horizons=steps,best_epoch=ck['best_epoch'],
                                         training_loop_and_final_evaluation_seconds=time.perf_counter()-started))
    pd.DataFrame([dict(case=name,seed=seed,**pooled)]).to_csv(folder/'summary_results.csv',index=False)
    print(f'COMPLETE {name}/{seed}: observed RMSE={pooled["node_rmse"]:.4f}',flush=True)


def aggregate(root,plan):
    rows=[]
    for case in CASES:
        for seed in SEEDS:
            f=root/f'seed_{seed}'/case['name']
            if not (f/'completed.json').exists():continue
            result=common.read(f/'metrics.json')
            rows.append(dict(case=case['name'],seed=seed,**result['pooled']))
    if not rows:return
    common.save(root/'all_completed_metrics.json',rows)
    frame=pd.DataFrame(rows);summary=[]
    for case,g in frame.groupby('case',sort=False):
        row=dict(case=case,seed_count=len(g))
        for key in [k for k in g if k.startswith(('node_','farm_'))]:
            row[key+'_mean']=float(g[key].mean())
            row[key+'_sd']=float(g[key].std()) if len(g)>1 else None
        summary.append(row)
    common.save(root/'summary.json',summary)
    if len(rows)==18:
        lines=['固定延迟与物理延迟、短历史与长历史：全部18组已完成。',
               '种子84/101/1234；原始有效测试标签；六步合并指标；均值±样本标准差。',
               'case | node RMSE(kW) | node MAE(kW) | farm RMSE(kW)']
        for r in summary:lines.append(r['case']+' | '+' | '.join(f'{r[k+"_mean"]:.3f}±{r[k+"_sd"]:.3f}' for k in ['node_rmse','node_mae','farm_rmse']))
        (root/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8')


def prepare_and_launch():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4)
    old=common.read(OLD/'plan.json');cache=Path(old['dataset_cache'])
    assert common.sha(OLD/'source_snapshot.py')==old['source_sha256']
    for d in old['data_files']:assert common.sha(Path(d['path']))==d['sha256']
    root=ROOT/('延迟历史窗口对照_84_101_1234_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    root.mkdir(exist_ok=False);(root/'data').mkdir()
    shutil.copy2(OLD/'source_snapshot.py',root/'source_snapshot.py')
    shutil.copy2(Path(__file__),root/'runner.py');shutil.copy2(Path(common.__file__),root/'runner_common.py')
    b=common.load_module(root);cfg=b.Config(**old['config']);df=b.load_sdwpf_long(cfg)
    turbines=common.read(cache/'ready.json')['turbines']
    rawpower=df.pivot_table(index='time',columns='turbine',values='power',aggfunc='mean').reindex(columns=turbines).sort_index()
    times,ws,wd,yaw,power=b.make_wide_matrices(df,turbines);del df
    assert rawpower.index.equals(times)
    features=np.stack([b.build_local_node_features(ws[t],wd[t],yaw[t],power[t],cfg) for t in range(len(times))])
    ntrain=len(np.load(cache/'train_Y_time.npy',mmap_mode='r'))
    train=np.lib.stride_tricks.sliding_window_view(features,3,axis=0)[:ntrain].transpose(0,3,1,2).copy()
    mu=train.reshape(-1,6).mean(0,keepdims=True);std=train.reshape(-1,6).std(0,keepdims=True)+1e-6
    del train
    normalized=(features-mu.reshape(1,1,6))/std.reshape(1,1,6)
    cached=np.load(cache/'train_X_time.npy',mmap_mode='r')
    for s in [0,1234,ntrain-1]:np.testing.assert_array_equal(normalized[s:s+3],cached[s])
    np.save(root/'data/features.npy',normalized)
    np.save(root/'data/times.npy',times.to_numpy())
    np.save(root/'data/normalization_mean.npy',mu);np.save(root/'data/normalization_std.npy',std)
    ty=np.load(cache/'test_times.npy').astype('datetime64[ns]');ix=times.get_indexer(ty.reshape(-1)).reshape(ty.shape)
    observed=rawpower.to_numpy()[ix].transpose(0,2,1).astype(np.float32)
    observed[~np.isfinite(observed)|(observed<0)]=np.nan
    np.save(root/'data/test_observed_kw.npy',observed)
    assert int(np.isfinite(observed).sum())==3753533
    common.save(root/'data/ready.json',dict(source_sha256=old['source_sha256'],
          files={p.name:common.sha(p) for p in (root/'data').glob('*.npy')},
          normalization_exactly_matches_original_L3_cache=True,observed_target_count=3753533,
          train_windows=ntrain-10,val_windows=7516,test_windows=7517))
    plan=copy.deepcopy(old);plan.update(workspace=str(ROOT),seeds=SEEDS,models=[c['name'] for c in CASES],
          cases=CASES,total=18,concurrency=2,runner_sha256=common.sha(root/'runner.py'),
          helper_sha256=common.sha(root/'runner_common.py'),data_ready_sha256=common.sha(root/'data/ready.json'),
          design='Full 3x2 factorial; all cases newly trained; common origins >=12, identical graph weights/topology/roles, fixed lag feature scaling by2, identical normalization, cap2, no test-driven selection',
          training='Original filled-label MSE+0.13 farm mean MSE; validation original filled-label pooled node RMSE; AdamW lr5e-4 wd1e-5, batch30,clip2,max50,patience8',
          evaluation='Raw finite nonnegative observed targets only; complete134-turbine farm targets; pooled and h1-h6',
          limitations='Retained original clock contains gaps; 10/20min describe nominal index lags. Physical delays are implementation proxies with3m/s speed floor; long history gives context, not recovery of >cap2 delay edges.')
    common.save(root/'plan.json',plan)
    # Preflight exact physical path and fixed-delay feature consistency, all six architectures.
    sample=Windows(root,'val',3)[0];x,y,n,e,m,r=[torch.from_numpy(a).unsqueeze(0) for a in sample]
    model=b.make_model(cfg,'full',6,n_turbines=134).eval()
    with torch.no_grad():
        assert torch.isfinite(model(x,n,e,m,r)).all()
        for strategy,v in [('fixed10',.5),('fixed20',1.)]:
            fixed=adjusted(e,strategy);assert torch.all(fixed[...,6][e[...,3]>0]==v)
            torch.testing.assert_close(fixed[...,:6],e[...,:6])
    counts=[]
    for case in CASES:
        cc=copy.deepcopy(cfg);cc.temporal_window=case['L'];mm=b.make_model(cc,'full',6,n_turbines=134)
        counts.append(sum(p.numel() for p in mm.parameters()))
    assert len(set(counts))==1
    common.save(root/'preflight.json',dict(status='passed',parameters_all_cases=counts,
          fixed_delay_encoding_consistent=True,L3_cached_normalization_exact=True,all_test_targets_aligned=True))
    (root/'RUN_INFO.txt').write_text('六配置×种子84、101、1234=18组全新训练；同训练/验证/测试起报与同图结构。\n'
        '延迟：physical/fixed10/fixed20；历史长度3/13。延迟检索上限始终2步，避免同时改变历史与传播上限。\n'
        '训练和验证保持前轮原脚本口径，测试只评原始有效标签；不能称为全流程未填补标签训练。\n'
        '当前保留时间轴存在缺口，延迟为名义采样索引含义，不能冒充严格连续物理时钟的验证。\n'
        '完成后RESULTS.txt和summary.json自动汇总。中途summary的seed_count不足3时不是最终结果。\n',encoding='utf-8')
    env=dict(os.environ,PYTHONUTF8='1',PYTHONUNBUFFERED='1')
    with (root/'controller.log').open('ab',buffering=0) as log:
        p=subprocess.Popen([sys.executable,'-u',str(root/'runner.py'),'--root',str(root)],cwd=root,
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            creationflags=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP)
    common.save(ROOT/'_comparison_multiseed_build/launched_delay_window.json',dict(directory=str(root),launcher_pid=p.pid,total=18,cases=CASES,seeds=SEEDS))
    print('LAUNCHED',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--seed',type=int);p.add_argument('--model');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.launch:prepare_and_launch()
    elif a.model:worker(a.root.resolve(),a.seed,a.model)
    else:
        common.verify=verify;common.aggregate=aggregate;common.controller(a.root.resolve())
