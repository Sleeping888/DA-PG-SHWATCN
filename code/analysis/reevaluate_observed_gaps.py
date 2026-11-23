"""Fixed-checkpoint evaluation on unfilled observed targets and input-gap strata."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'对比实验_1234_21_42_84_20261119_164746'
NEW=ROOT/'对比实验_101_20261120_223936'
OUT=ROOT/'原始有效标签与缺失分层_84_101_1234_20261122'
SEEDS=[84,101,1234]
BINS=['无缺失','0–30min','30–60min','60–180min','180–1440min','>1440min']

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(4*1024**2),b''):h.update(a)
    return h.hexdigest()

def score(y,p,capacity):
    if not len(y):return dict(count=0,rmse=None,mae=None,nrmse=None,nmae=None,r2=None)
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    sse=np.sum((y-p)**2);mae=np.mean(np.abs(y-p));rmse=np.sqrt(sse/len(y));sst=np.sum((y-y.mean())**2)
    return dict(count=len(y),rmse=float(rmse),mae=float(mae),nrmse=float(100*rmse/capacity),
                nmae=float(100*mae/capacity),r2=float(1-sse/sst) if sst>0 else None)

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);OUT.mkdir(exist_ok=True)
    plan=read(OLD/'plan.json');other=read(NEW/'plan.json')
    assert plan['source_sha256']==other['source_sha256']
    for r in [OLD,NEW]:assert read(r/'status.json')['status']=='complete'
    for d in plan['data_files']:assert sha(Path(d['path']))==d['sha256']
    spec=importlib.util.spec_from_file_location('observed_eval_source',OLD/'source_snapshot.py')
    b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
    assert sha(OLD/'source_snapshot.py')==plan['source_sha256']
    cache=Path(plan['dataset_cache']);meta=read(cache/'ready.json')
    arrays={k:np.load(cache/f'test_{k}.npy',mmap_mode='r') for k in
            ['X_time','Y_matrix','nbr_idx','edge_attr','edge_mask','role','times']}
    cfg=b.Config();df=pd.read_csv(cfg.data_csv,usecols=['TurbID','Tmstamp','Wspd','Wdir','Ndir','Patv'])
    stamp=pd.to_datetime(df.Tmstamp).to_numpy(dtype='datetime64[ns]')
    clock=pd.date_range(stamp.min(),stamp.max(),freq='10min')
    ii=clock.get_indexer(stamp);jj=df.TurbID.to_numpy(dtype=int)-1
    assert (ii>=0).all() and ((jj>=0)&(jj<134)).all()
    assert not pd.MultiIndex.from_arrays([ii,jj]).duplicated().any()
    T=len(clock);N=134;raw={}
    for field in ['Wspd','Wdir','Ndir','Patv']:
        a=np.full((T,N),np.nan,dtype=np.float64);a[ii,jj]=df[field].to_numpy(dtype=np.float64);raw[field]=a
    del df,stamp,ii,jj
    raw_y=raw['Patv'];observed=np.isfinite(raw_y)&(raw_y>=0)
    maxage=np.zeros((T,N),np.int32);maxgap=np.zeros_like(maxage)
    field_records={};row=np.arange(T)[:,None]
    for field,mat in raw.items():
        valid=np.isfinite(mat)
        if field=='Wspd':valid &= (mat>=0)&(mat<=40)
        if field=='Patv':valid &= mat>=0
        # The original loader wraps angles modulo 360 before cleaning.
        prev=np.maximum.accumulate(np.where(valid,row,-1),axis=0)
        age=np.where(valid,0,np.where(prev>=0,row-prev,row+1)).astype(np.int32)*10
        maxage=np.maximum(maxage,age)
        lengths=np.zeros((T,N),np.int32);run_count=0
        for j in range(N):
            bad=~valid[:,j];change=np.diff(np.r_[False,bad,False].astype(np.int8))
            starts=np.flatnonzero(change==1);ends=np.flatnonzero(change==-1)
            for start,end in zip(starts,ends):lengths[start:end,j]=(end-start)*10
            run_count+=len(starts)
        maxgap=np.maximum(maxgap,lengths)
        field_records[field]=dict(invalid_cells=int((~valid).sum()),gap_runs=run_count,
                  longest_gap_minutes=int(lengths.max()),max_elapsed_missing_minutes=int(age.max()))
    target_times=arrays['times'].astype('datetime64[ns]');target_idx=clock.get_indexer(target_times.reshape(-1)).reshape(target_times.shape)
    assert (target_idx>=0).all()
    y=raw_y[target_idx].transpose(0,2,1).copy();mask=observed[target_idx].transpose(0,2,1)
    assert y.shape==arrays['Y_matrix'].shape
    legacy_times=np.load(cache/'pred_times.npy').astype('datetime64[ns]')
    # Cached prediction origin is two positions before h1 on retained clock.
    # Recover the exact three local input timestamps without assuming a regular original clock.
    power_times=pd.DatetimeIndex(clock[observed.any(axis=1)])
    assert len(power_times)==50116
    all_origins=power_times[:-6][2:]
    offset=meta['samples']['train']+meta['samples']['val']
    origin_pos=np.arange(2+offset,2+offset+len(y))
    local_times=np.stack([power_times.to_numpy()[origin_pos-i] for i in [2,1,0]],axis=1)
    assert np.array_equal(target_times[:,0],power_times.to_numpy()[origin_pos+1])
    local_idx=clock.get_indexer(local_times.reshape(-1)).reshape(local_times.shape)
    strata_age=np.max(maxage[local_idx],axis=1)
    strata_gap=np.max(maxgap[local_idx],axis=1)
    class_age=np.searchsorted([0,30,60,180,1440],strata_age,side='left')
    class_gap=np.searchsorted([0,30,60,180,1440],strata_gap,side='left')
    coverage=[]
    for kind,c in [('full_gap',class_gap),('elapsed_age',class_age)]:
        for group in range(6):
            m=mask&(c==group)[...,None]
            coverage.append(dict(kind=kind,group=BINS[group],observed_target_pairs=int(m.sum()),
                                 turbine_origin_pairs=int((c==group).sum())))
        assert sum(z['observed_target_pairs'] for z in coverage if z['kind']==kind)==int(mask.sum())
    save(OUT/'coverage.json',dict(possible_target_pairs=int(mask.size),valid_target_pairs=int(mask.sum()),
           excluded_target_pairs=int((~mask).sum()),complete_farm_pairs=int(mask.all(1).sum()),
           valid_targets_above_original_2250kw_clip=int((mask&(y>2250)).sum()),strata=coverage,
           fields=field_records))
    print('RAW TARGET COVERAGE',int(mask.sum()),'/',mask.size,flush=True)
    records=[];replays=[]
    with torch.inference_mode():
        for mode in plan['models']:
            for seed in SEEDS:
                folder=(NEW if seed==101 else OLD)/f'seed_{seed}'/mode
                done=read(folder/'completed.json');assert done['summary_sha256']==sha(folder/'summary_results.csv')
                ck=torch.load(folder/'checkpoint_best.pt',map_location='cpu',weights_only=False)
                cc=b.Config(**ck['config']);state=ck['model_state_dict']
                if mode=='stv_pgnn_adapted':
                    cc._stv_context=dict(xyz=state['xyz'].numpy(),mean=state['feature_mean'].numpy(),std=state['feature_std'].numpy())
                model=b.make_model(cc,mode,6,n_turbines=134).cuda().eval();model.load_state_dict(state,strict=True)
                predictions=[];physical=mode in {'full','stgcn'}
                for start in range(0,len(y),30):
                    stop=min(start+30,len(y));batch=[torch.from_numpy(arrays['X_time'][start:stop].copy()).float().cuda()]
                    for key in ['nbr_idx','edge_attr','edge_mask','role']:
                        a=arrays[key][start:stop].copy()
                        if not physical:a.fill(0)
                        t=torch.from_numpy(a)
                        batch.append((t.float() if key in {'edge_attr','role'} else t).cuda())
                    predictions.append(model(*batch).cpu().numpy())
                pred=np.clip(np.concatenate(predictions),0,1.5)*cc.rated_power_kw
                oldfirst=pd.read_csv(folder/f'test_predictions_{mode}.csv',nrows=804).pred_kw.to_numpy().reshape(6,134).T
                np.testing.assert_allclose(pred[0],oldfirst,rtol=1e-5,atol=.002)
                replays.append(dict(model=mode,seed=seed,checkpoint_sha256=sha(folder/'checkpoint_best.pt'),
                                    first_origin_max_diff_kw=float(np.abs(pred[0]-oldfirst).max())))
                # Independent compatibility check against original reported node AVG.
                legacy_y=np.asarray(arrays['Y_matrix'])*cc.rated_power_kw
                legacy_rmse=score(legacy_y.ravel(),pred.ravel(),1500)['rmse']
                exported=pd.read_csv(folder/'summary_results.csv').iloc[0].test_node_avg_rmse
                assert abs(legacy_rmse-exported)<.003,(mode,seed,legacy_rmse,exported)
                for h in range(7):
                    yy,pp,mm=(y,pred,mask) if h==0 else (y[...,h-1],pred[...,h-1],mask[...,h-1])
                    for level in ['node','farm']:
                        if level=='node':sy,sp=yy[mm],pp[mm];cap=1500
                        else:
                            complete=mm.all(1);sy=yy.sum(1)[complete];sp=pp.sum(1)[complete];cap=201000
                        records.append(dict(model=mode,seed=seed,scope='observed',level=level,horizon=h,
                                            **score(sy,sp,cap)))
                    # Local input-gap strata; farm evaluated only as a whole, never summing a subset of turbines.
                    for kind,c in [('full_gap',class_gap),('elapsed_age',class_age)]:
                        for group in range(6):
                            use=mm&((c==group)[...,None] if h==0 else (c==group))
                            records.append(dict(model=mode,seed=seed,scope=kind,group=BINS[group],level='node',horizon=h,
                                                **score(yy[use],pp[use],1500)))
                save(OUT/'per_horizon_metrics.json',records)
                print(f'PASS {mode}/{seed} observed pooled RMSE '+str(next(r['rmse'] for r in records[-98:] if r['scope']=='observed' and r['level']=='node' and r['horizon']==0)),flush=True)
                del model,pred,predictions,batch
    f=pd.DataFrame(records);keys=['rmse','mae','nrmse','nmae','r2']
    per=[]
    for cols,g in f[f.horizon.eq(0)].groupby(['model','seed','scope','level','group'],dropna=False,sort=False):
        model,seed,scope,level,group=cols
        r=dict(model=model,seed=int(seed),scope=scope,level=level,group=None if pd.isna(group) else group,
               valid_horizons=int(g.rmse.notna().sum()),target_pairs=int(g['count'].sum()))
        for key in keys:r[key]=None if g[key].isna().all() else float(g[key].mean())
        per.append(r)
    save(OUT/'per_seed_pooled_metrics.json',per)
    p=pd.DataFrame(per);summary=[]
    for cols,g in p.groupby(['model','scope','level','group'],dropna=False,sort=False):
        model,scope,level,group=cols;r=dict(model=model,scope=scope,level=level,group=None if pd.isna(group) else group,seeds=len(g))
        for key in keys:
            r[key+'_mean']=None if g[key].isna().all() else float(g[key].mean())
            r[key+'_sd']=None if g[key].notna().sum()<2 else float(g[key].std(ddof=1))
        summary.append(r)
    save(OUT/'summary.json',summary);save(OUT/'checkpoint_replays.json',replays)
    lines=['原始有效功率标签重新评估；84、101、1234三个种子；固定检查点，无重新训练。',
       '有效标签=原始Patv有限且非负；不前向填补、不置零、不裁剪有效观测。全场仅在134台原始标签全部有效时评估。',
       '本报告主表为六步合并指标（pooled），再对3种子求均值±样本标准差。原脚本AVG同样是pooled。此前聊天中将其说成逐步算术平均的描述有误，原列出的数值不变。horizon=0表示合并，horizon=1..6表示逐步。',
       '缺口定义：连续10分钟物理时钟下，每个字段连续缺失/无效段的完整长度；每个目标风机按其原3点局部输入窗口中四字段的最大缺口长度分层。',
       '完整缺口长度仅用于事后分层，可能使用之后的恢复信息，不作为预测输入。另导出只使用起报时已知携带年龄的elapsed_age分层。',
       '字段有效规则匹配原输入清洗：Wspd∈[0,40]，角度为有限值（原实现先模360），Patv有限且非负。缺失整行时刻也计入连续缺口长度。',
       '分层只解释本机局部输入缺失，未覆盖所有上游消息源的缺失。固定检查点意味着历史训练中的填补标签影响仍然存在，本次只修正测试标签评分。',
       '各层目标数量不同，比较层间误差属于观察性稳健性描述，不是人为缺失注入实验的因果效应。']
    for level in ['node','farm']:
        lines+=['',level+' observed pooled:','model | RMSE(kW) | MAE(kW) | nRMSE(%) | nMAE(%) | R2']
        for r in sorted([r for r in summary if r['scope']=='observed' and r['level']==level],key=lambda r:r['rmse_mean']):
            lines.append(r['model']+' | '+' | '.join(f'{r[k+"_mean"]:.4f}±{r[k+"_sd"]:.4f}' for k in keys))
    for kind in ['full_gap','elapsed_age']:
        lines+=['',f'Full input strata: {kind}']
        for group in BINS:
            r=next(r for r in summary if r['model']=='full' and r['scope']==kind and r['group']==group)
            count=next(r['observed_target_pairs'] for r in coverage if r['kind']==kind and r['group']==group)
            lines.append(group+f' | n={count} | '+' | '.join('n.a.' if r[k+'_mean'] is None else f'{r[k+"_mean"]:.4f}±{r[k+"_sd"]:.4f}' for k in ['rmse','mae','r2']))
    (OUT/'重新评估结果与口径.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'completed.json',dict(status='passed',models=9,seeds=SEEDS,runs=27,
         source_sha256=plan['source_sha256'],runner_sha256=sha(Path(__file__)),new_training=False))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
