"""Audit delay/filling frequencies for the actual November comparison protocol."""
from pathlib import Path
import importlib.util
import json
import sys
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'对比实验_1234_21_42_84_20261119_164746'
OUT=ROOT/'延迟与填充频率_当前对比模型_20261122'

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(exist_ok=True)
    plan=json.loads((RUN/'plan.json').read_text(encoding='utf-8'))
    source=RUN/'source_snapshot.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_sha256']
    spec=importlib.util.spec_from_file_location('delay_audit_original',source)
    b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
    cfg=b.Config(**plan['config']);cache=Path(plan['dataset_cache'])
    meta=json.loads((cache/'ready.json').read_text(encoding='utf-8'))
    turbines=meta['turbines'];N=len(turbines)
    df=b.load_sdwpf_long(cfg)
    frames={};missing={};fields={}
    for name,field,low,high in [('Wspd','ws',0,40),('Wdir','wd',0,360),('Ndir','yaw',0,360),('Patv','power',0,None)]:
        p=df.pivot_table(index='time',columns='turbine',values=field,aggfunc='mean').reindex(columns=turbines).sort_index()
        p=p.mask(p<low)
        if high is not None:p=p.mask(p>high)
        m=p.isna().to_numpy()
        leading=(p.ffill().isna()).to_numpy()
        missing[name]=m
        fields[name]=dict(total_cells=int(m.size),filled_cells=int(m.sum()),filled_percent=float(100*m.mean()),
                          leading_zero_filled_cells=int(leading.sum()))
        frames[name]=p
        print(name,fields[name],flush=True)
    times=pd.DatetimeIndex(frames['Patv'].index)
    assert all(f.index.equals(times) for f in frames.values())
    ws=frames['Wspd'].ffill().fillna(0).to_numpy(dtype=np.float32)
    wd=frames['Wdir'].ffill().fillna(0).to_numpy(dtype=np.float32)%360
    yaw=frames['Ndir'].ffill().fillna(0).to_numpy(dtype=np.float32)%360
    power=frames['Patv'].ffill().fillna(0).to_numpy(dtype=np.float32)
    del df,frames
    any_missing=np.logical_or.reduce(list(missing.values()))
    x,y=b.load_layout(cfg,turbines)
    results={};all_seconds=[];global_origin=2
    for split in ['train','val','test']:
        g={k:np.load(cache/f'{split}_{k}.npy',mmap_mode='r') for k in ['nbr_idx','edge_attr','edge_mask']}
        S=len(g['nbr_idx']);origin=np.arange(global_origin,global_origin+S)
        assert np.array_equal(np.load(cache/f'{split}_times.npy')[:,0],times.to_numpy()[origin+1])
        for s in [0,S//2,S-1]:
            t=origin[s]
            _,nbr,e,m,_=b.build_time_graph_sample(ws[t],wd[t],yaw[t],power[t],x,y,cfg,'full')
            np.testing.assert_array_equal(nbr,g['nbr_idx'][s])
            np.testing.assert_array_equal(e,g['edge_attr'][s])
            np.testing.assert_array_equal(m,g['edge_mask'][s])
        counts=dict(real_edges=0,raw_delay_gt_1step=0,raw_delay_gt_2steps=0,
                    encoded_delay_uses_oldest_point=0,retrieved_endpoint_has_filled_input=0,
                    retrieval_before_start=0,delay_speed_floor_edges=0,
                    timestamp_retrieval_gt20min_edges=0,origins_with_any_oldest_retrieval=0,
                    origins_with_any_filled_retrieval=0,origins=S)
        for start in range(0,S,256):
            ix=np.arange(start,min(start+256,S));t=origin[ix]
            src=g['nbr_idx'][ix].astype(np.int64);edge=g['edge_attr'][ix].astype(np.float32)
            real=g['edge_mask'][ix]&(edge[...,3]>0)
            angle=np.deg2rad(np.mod(np.rad2deg(np.arctan2(np.sin(np.deg2rad(wd[t])).mean(1),
                                                          np.cos(np.deg2rad(wd[t])).mean(1)))+180,360))
            axial=(x[None,:,None]-x[src])*np.cos(angle)[:,None,None]+(y[None,:,None]-y[src])*np.sin(angle)[:,None,None]
            speed=ws[t].mean(1)
            raw=axial/(np.maximum(speed,3)[:,None,None]*600+1e-6)
            delay=np.clip(edge[...,6],0,1)*2
            floor=np.floor(delay).astype(int);ceil=np.ceil(delay).astype(int)
            t0=t[:,None,None]-floor;t1=t[:,None,None]-ceil
            # The ceil endpoint has nonzero weight only for a fractional lag.
            use_other=delay>floor
            filled=any_missing[t0,src]|(use_other&any_missing[t1,src])
            before=(t0<0)|(use_other&(t1<0))
            oldest=delay>1
            elapsed0=(times.to_numpy()[t[:,None,None]]-times.to_numpy()[t0])/np.timedelta64(1,'s')
            elapsed1=(times.to_numpy()[t[:,None,None]]-times.to_numpy()[t1])/np.timedelta64(1,'s')
            beyond=(elapsed0>1200)|(use_other&(elapsed1>1200))
            counts['real_edges']+=int(real.sum())
            for key,mask in [('raw_delay_gt_1step',raw>1),('raw_delay_gt_2steps',raw>2),
                ('encoded_delay_uses_oldest_point',oldest),('retrieved_endpoint_has_filled_input',filled),
                ('retrieval_before_start',before),('delay_speed_floor_edges',np.broadcast_to((speed<3)[:,None,None],real.shape)),
                ('timestamp_retrieval_gt20min_edges',beyond)]:
                counts[key]+=int((mask&real).sum())
            counts['origins_with_any_oldest_retrieval']+=int((oldest&real).any((1,2)).sum())
            counts['origins_with_any_filled_retrieval']+=int((filled&real).any((1,2)).sum())
            all_seconds.append(raw[real].astype(np.float64)*600)
        results[split]=counts;global_origin+=S
        print(split,counts,flush=True)
    total={k:sum(v[k] for v in results.values()) for k in results['train']}
    seconds=np.concatenate(all_seconds)
    report=dict(scope='Actual cached comparison samples, same frozen source as seeds 84/101/1234; real nonzero physical edges only',
                source_sha256=plan['source_sha256'],samples=global_origin-2,timestamps=len(times),fields=fields,
                splits=results,total=total,percentages={k:100*v/total['real_edges'] for k,v in total.items()
                    if k not in ['real_edges','origins','origins_with_any_oldest_retrieval','origins_with_any_filled_retrieval']},
                delay_seconds=dict(mean=float(seconds.mean()),p50=float(np.median(seconds)),p95=float(np.percentile(seconds,95)),
                                   p99=float(np.percentile(seconds,99)),max=float(seconds.max())),
                longer_window_configured_percent=0,delay_null_imputation_mechanism=False,
                nominal_tcn_receptive_field=5,history_points=3,
                timestamp_gaps=int(np.count_nonzero(np.diff(times.to_numpy())!=np.timedelta64(10,'m'))),
                limitations=['3 m/s floor is an empirical proxy, not a measured transport-speed bound',
                             'Original comparison clock omits fully missing timestamps',
                             'Filled-input exposure refers to directly interpolated source timestamps; TCN context can include additional filled cells',
                             'Fractional delay interpolation is not missing-value imputation',
                             'No dynamic long-window switch is implemented; 0% clipping does not mean no TCN convolution padding'])
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['实际对比模型的延迟与填充统计',f'样本数{report["samples"]}；有效物理边实例{total["real_edges"]}。',
           '分母为各样本保留的非零物理边实例，不包括dummy self和空邻居槽；跨随机种子重复使用相同输入，不乘以种子数。',
           json.dumps(report['percentages'],ensure_ascii=False,indent=2),
           '空值前向填充比例（字段单元）：'+json.dumps(fields,ensure_ascii=False),
           '当前固定L=3、延迟cap=2、步长10 min；没有按事件动态启用更长历史窗口。',
           '纯因延迟缺失而填零的机制不存在；延迟过长时采用截断而非NaN填充。TCN卷积左补零与这两者不同。',
           '原时钟存在非10min间隔；按数组索引可检索不代表按实际物理时钟同样可检索。详见timestamp_retrieval_gt20min_edges。',
           '全部统计来自当前对比实验缓存和同版原始数据重放，不套用修正连续时钟研究中的22.08%数字。']
    (OUT/'统计口径.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
