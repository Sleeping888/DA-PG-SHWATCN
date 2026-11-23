"""Build an uploadable, checksummed archive of actual code, indices and forecasts."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import shutil
import sys
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
FOUR=ROOT/'对比实验_1234_21_42_84_20261119_164746'
ONE=ROOT/'对比实验_101_20261120_223936'
DELAY=ROOT/'延迟历史窗口对照_84_101_1234_20261122_152532'
OUT=ROOT/'论文上传材料_代码配置索引与完整预测_20261123'
PRIMARY=[84,101,1234]

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for a in iter(lambda:f.read(4*1024**2),b''):h.update(a)
    return h.hexdigest()
def cp(a,b):
    b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a,b)

def split_files(times,meta):
    count=sum(meta['samples'].values());allorig=np.arange(2,2+count,dtype=np.int64)
    starts=[0,meta['samples']['train'],meta['samples']['train']+meta['samples']['val'],count]
    records={}
    for study,L,minorigin in [('comparison',3,2),('delay_L3',3,12),('delay_L13',13,12)]:
        data={};notes={}
        for i,split in enumerate(['train','val','test']):
            sample=np.arange(starts[i],starts[i+1]);origin=allorig[sample];keep=origin>=minorigin
            sample=sample[keep];origin=origin[keep]
            history=origin[:,None]-np.arange(L-1,-1,-1)[None,:]
            target=origin[:,None]+np.arange(1,7)[None,:]
            for k,v in dict(sample_index=sample,origin_index=origin,history_indices=history,target_indices=target,
                            origin_time=times[origin],history_times=times[history],target_times=times[target]).items():
                data[split+'_'+k]=v
            notes[split]=dict(samples=len(sample),first_origin=str(times[origin[0]]),last_origin=str(times[origin[-1]]),
                             first_origin_index=int(origin[0]),last_origin_index=int(origin[-1]))
        for left,right in [('train','val'),('val','test')]:
            a=np.unique(data[left+'_target_indices']);b=np.unique(data[right+'_target_indices'])
            notes[left+'_'+right+'_shared_target_timestamps']=int(len(np.intersect1d(a,b)))
        dest=OUT/'splits'/f'{study}_indices.npz';dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,**data);records[study]=notes
    save(OUT/'splits/definitions.json',dict(index_base=0,index_clock='data/retained_times.npy',
         comparison=dict(L=3,first_origin=2),delay=dict(L=[3,13],first_origin=12),
         studies=records,important='Actual overlapping multi-horizon targets are preserved; no claim of purged non-overlapping split.'))

def raw_targets(plan,target_times):
    unique=np.unique(target_times);matrix=np.full((len(unique),134),np.nan,np.float64)
    seen=np.zeros_like(matrix,dtype=bool);clock=pd.DatetimeIndex(unique)
    for chunk in pd.read_csv(plan['config']['data_csv'],usecols=['TurbID','Tmstamp','Patv'],chunksize=400000):
        tt=pd.to_datetime(chunk.Tmstamp).to_numpy(dtype='datetime64[ns]');ix=clock.get_indexer(tt);keep=ix>=0
        ii=ix[keep];jj=chunk.TurbID.to_numpy(dtype=int)[keep]-1
        assert not seen[ii,jj].any();seen[ii,jj]=True;matrix[ii,jj]=chunk.Patv.to_numpy(dtype=np.float64)[keep]
    ii=clock.get_indexer(target_times.reshape(-1)).reshape(target_times.shape)
    raw=matrix[ii].transpose(0,2,1);valid=np.isfinite(raw)&(raw>=0)
    assert int(valid.sum())==3753533
    return raw,valid

def convert_csv(path,expected_true,target_times,turbines):
    S,N,H=expected_true.shape;pred=np.empty(S*N*H,np.float32);pos=0
    truths=expected_true.transpose(0,2,1).reshape(-1)
    tt=np.repeat(target_times.reshape(-1),N)
    for chunk in pd.read_csv(path,chunksize=350000):
        stop=pos+len(chunk);ix=np.arange(pos,stop)
        assert np.array_equal(chunk.horizon.to_numpy(),(ix//N)%H+1)
        assert np.array_equal(chunk.turbine.to_numpy(),np.asarray(turbines,dtype=object)[ix%N])
        assert np.array_equal(pd.to_datetime(chunk.target_time).to_numpy(dtype='datetime64[ns]'),tt[pos:stop])
        np.testing.assert_allclose(chunk.true_kw.to_numpy(np.float32),truths[pos:stop],rtol=0,atol=1e-4)
        pred[pos:stop]=chunk.pred_kw.to_numpy(np.float32);pos=stop
    assert pos==len(pred) and np.isfinite(pred).all()
    return pred.reshape(S,H,N).transpose(0,2,1)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    if OUT.exists():raise RuntimeError('Output folder already exists; do not overwrite an upload release')
    OUT.mkdir();plan=read(FOUR/'plan.json');meta=read(Path(plan['dataset_cache'])/'ready.json');cache=Path(plan['dataset_cache'])
    for root in [FOUR,ONE,DELAY]:assert read(root/'status.json')['status']=='complete'
    assert read(ONE/'plan.json')['source_sha256']==plan['source_sha256']==read(DELAY/'plan.json')['source_sha256']
    for d in plan['data_files']:assert sha(Path(d['path']))==d['sha256']
    times=np.load(DELAY/'data/times.npy').astype('datetime64[ns]');np.save(OUT/'retained_times.tmp.npy',times)
    cp(OUT/'retained_times.tmp.npy',OUT/'data/retained_times.npy');(OUT/'retained_times.tmp.npy').unlink()
    split_files(times,meta)
    target_times=np.load(cache/'test_times.npy').astype('datetime64[ns]')
    asrun=np.asarray(np.load(cache/'test_Y_matrix.npy',mmap_mode='r'))*1500
    raw,valid=raw_targets(plan,target_times)
    source_observed=np.load(DELAY/'data/test_observed_kw.npy')
    np.testing.assert_array_equal(np.isfinite(source_observed),valid)
    np.testing.assert_allclose(source_observed[valid],raw[valid],rtol=1e-6,atol=.002)
    origin_idx=np.arange(2+meta['samples']['train']+meta['samples']['val'],2+sum(meta['samples'].values()))
    origin_time=times[origin_idx];turbines=meta['turbines'];turbine_id=np.arange(1,135,dtype=np.int16)
    np.savez_compressed(OUT/'data/shared_test_targets.npz',true_kw_raw=raw,true_kw_as_run=asrun,
                        observed_flag=valid,origin_time=origin_time,target_time=target_times,turbine_id=turbine_id)
    print('Prepared exact splits and shared raw/as-run labels',flush=True)
    for d in plan['data_files']:cp(Path(d['path']),OUT/'data/raw'/Path(d['path']).name)
    for name in ['normalization_mean.npy','normalization_std.npy','features.npy','ready.json']:
        cp(DELAY/'data'/name,OUT/'data/delay_preprocessing'/name)
    cp(DELAY/'data/test_observed_kw.npy',OUT/'data/delay_preprocessing/test_observed_kw.npy')
    cp(DELAY/'data/times.npy',OUT/'data/delay_preprocessing/times.npy')
    cp(cache/'ready.json',OUT/'data/original_cache_ready.json')
    for label,root in [('comparison_four_seeds',FOUR),('comparison_seed101',ONE),('delay_window',DELAY)]:
        for name in ['source_snapshot.py','runner.py','runner_common.py','plan.json','preflight.json','status.json']:
            if (root/name).exists():cp(root/name,OUT/'code/as_run'/label/name)
    for name in ['reevaluate_observed_gaps.py','audit_delay_usage.py','benchmark_efficiency.py']:
        cp(ROOT/'_comparison_multiseed_build'/name,OUT/'code/analysis'/name)
    cp(Path(__file__),OUT/'code/package_upload.py')
    cp(ROOT/'_comparison_multiseed_build/upload_read_predictions.py',OUT/'tools/read_predictions.py')
    cp(ROOT/'_comparison_multiseed_build/upload_verify.py',OUT/'tools/verify_package.py')
    cp(ROOT/'_comparison_multiseed_build/upload_rebuild.py',OUT/'tools/rebuild_preprocessing.py')
    runs=[]
    jobs=[]
    for root,seeds in [(FOUR,[1234,21,42,84]),(ONE,[101])]:
        for seed in seeds:
            for model in plan['models']:jobs.append(('comparison',root,seed,model))
    for seed in PRIMARY:
        for case in read(DELAY/'plan.json')['models']:jobs.append(('delay_window',DELAY,seed,case))
    for number,(study,root,seed,case) in enumerate(jobs,1):
        folder=root/f'seed_{seed}'/case;dest=OUT/'experiments'/study/f'seed_{seed}'/case
        done=read(folder/'completed.json');assert done['exit_code']==0 and sha(folder/'summary_results.csv')==done['summary_sha256']
        for path in folder.iterdir():
            if path.name in ['predictions.npz',f'test_predictions_{case}.csv']:continue
            if path.suffix in ['.json','.csv','.log','.pt']:cp(path,dest/path.name)
        if study=='comparison':
            source=folder/f'test_predictions_{case}.csv'
            pred=convert_csv(source,asrun,target_times,turbines)
        else:
            source=folder/'predictions.npz'
            with np.load(source,allow_pickle=False) as z:pred=z['pred_kw']
            assert pred.shape==asrun.shape and np.isfinite(pred).all()
        # Each run file is self-contained for inspecting all six forecast horizons.
        np.savez_compressed(dest/'forecasts_h1_h6.npz',pred_kw=pred,true_kw_raw=raw,
              true_kw_as_run=asrun,observed_flag=valid,origin_time=origin_time,target_time=target_times,turbine_id=turbine_id)
        with np.load(dest/'forecasts_h1_h6.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['pred_kw'],pred)
            assert np.array_equal(z['observed_flag'],valid)
        row=dict(study=study,seed=seed,case=case,primary_paper_seed=seed in PRIMARY,
                 path=str(dest.relative_to(OUT)).replace('\\','/'),shape=list(pred.shape),
                 original_prediction_path=str(source),original_prediction_sha256=sha(source),
                 raw_valid_target_pairs=int(valid.sum()),raw_complete_farm_pairs=int(valid.all(1).sum()))
        runs.append(row);save(OUT/'run_index.json',runs)
        print(f'PACKED {number}/{len(jobs)} {study}/{seed}/{case}',flush=True)
    reports=[('observed_evaluation',ROOT/'原始有效标签与缺失分层_84_101_1234_20261122'),
             ('delay_filling_audit',ROOT/'延迟与填充频率_当前对比模型_20261122'),
             ('efficiency',ROOT/'计算效率_84_101_1234_20261122'),
             ('figures',ROOT/'论文对比图_84_101_1234_20261122')]
    for label,root in reports:
        for p in root.iterdir():
            if not p.is_file() or p.name=='reference_preview.png':continue
            if p.suffix in ['.json','.txt','.py','.svg','.pdf','.png','.tiff']:cp(p,OUT/'reports'/label/p.name)
    for label,root in [('comparison_four_seeds',FOUR),('comparison_seed101',ONE),('delay_window',DELAY)]:
        for name in ['all_metrics_per_seed.csv','all_horizons_per_seed.csv','mean_std_by_horizon.csv',
                     'all_completed_metrics.json','summary.json','RESULTS.txt']:
            if (root/name).exists():cp(root/name,OUT/'reports'/label/name)
    save(OUT/'environment_packaging.json',dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,
          inspected_at=datetime.now().isoformat(),note='Inspected packaging environment; original hardware timing records are under reports/efficiency'))
    (OUT/'requirements.txt').write_text('numpy=='+np.__version__+'\npandas=='+pd.__version__+'\ntorch==2.4.1\nmatplotlib\n',encoding='utf-8')
    (OUT/'README.txt').write_text(README,encoding='utf-8')
    save(OUT/'release_info.json',dict(status='complete',comparison_runs=45,delay_window_runs=18,total_runs=63,
             primary_paper_seeds=PRIMARY,other_comparison_seeds=[21,42],source_sha256=plan['source_sha256'],
             target_shape=list(asrun.shape),raw_valid_target_pairs=int(valid.sum()),
             raw_complete_farm_pairs=int(valid.all(1).sum()),release_created_at=datetime.now().isoformat()))
    manifest=[]
    for p in sorted(OUT.rglob('*')):
        if p.is_file():manifest.append(dict(path=p.relative_to(OUT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    save(OUT/'manifest_sha256.json',manifest)
    print('Creating ZIP',flush=True)
    archive=OUT.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',allowZip64=True) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix in ['.npz','.png','.tiff'] else zipfile.ZIP_DEFLATED,compresslevel=3)
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip();assert bad is None,bad
        assert len(z.infolist())==len(manifest)+1
    record=dict(folder=str(OUT),archive=str(archive),archive_bytes=archive.stat().st_size,
          archive_sha256=sha(archive),file_count=len(manifest)+1,total_runs=63,zip_crc_check='passed')
    save(ROOT/'_comparison_multiseed_build/upload_package_completed.json',record)
    OUT.with_suffix('.zip.sha256.txt').write_text(record['archive_sha256']+'  '+archive.name+'\n',encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)

README='''UPLOAD RELEASE / 代码、实际配置、划分索引与完整六步预测

范围：同一源代码版本下45组模型对比（种子1234、21、42、84、101）和18组延迟/窗口对照（84、101、1234）。正文最近采用的三个种子84、101、1234在run_index.json中标记。额外种子保留以避免隐藏已有实验。历史修正协议的系数灵敏度不属于这一批次，未与本包混合。

1. code/as_run：实际执行的字节相同源码、运行适配器、原计划和原预检查。source_snapshot.py中load_sdwpf_long/make_wide_matrices是预处理；build_three_wake_matrices/build_topk_wake_neighbors/build_time_graph_sample是构图。原绝对路径保留用于审计，不代表其他机器上的目录。
2. experiments：每组实际config、训练history、最佳checkpoint、完成记录、日志及原指标。比较实验history_<model>.csv和延迟对照history.csv给出实际轮次。两者训练与验证都保留原填补标签流程；原始有效标签修正发生在测试评分阶段。
3. data/raw：实际输入CSV和布局CSV。data/retained_times.npy是原程序保留时间轴。data/delay_preprocessing包含实际标准化参数和标准化局部特征，features.npy形状[T,134,6]；normalization以原3点训练缓存精确重建，13点对照沿用它。
4. splits：零基索引NPZ，逐样本给出起报、历史输入及六个目标的索引和实际时间戳。comparison为原3点配置；delay_L3/delay_L13为公平对照，统一从retained index12开始。实际窗口数：比较train35075/val7516/test7517；延迟对照train35065/val7516/test7517。
5. 划分警示：原划分按样本窗口分段，边界附近多步目标时间有重叠，definitions.json记录交集数量。本包忠实保留实际划分，不能声称purged时间不重叠划分。原保留时间轴存在59处非10分钟间隔，10/20min为名义采样索引含义。

完整预测：每组forecasts_h1_h6.npz均包含[7517,134,6]数组。轴为起报样本、TurbID升序、h1..h6；每份共有6,043,668个预测数值，绝非抽样结果。
 pred_kw：实际模型预测，kW；comparison从原CSV逐行核对后无损转为float32，delay从原NPZ复制。
 true_kw_raw：原始功率观测，包括原始负值/NaN，不作前填、置零或裁剪。
 observed_flag：isfinite(true_kw_raw) & (true_kw_raw>=0)。原始有效标签评分必须使用该掩码，每组3,753,533个有效目标对。
 true_kw_as_run：原程序评分使用的填补/裁剪标签，kW，与原预测CSV真实值逐行核对。它不是原始有效观测，不能与true_kw_raw混称。
 origin_time：[7517]，起报时间；target_time：[7517,6]，真实目标时间；turbine_id：[134]，1..134。
 全场评分仅在某个起报/步长上134台observed_flag都为True时，对134台原始功率和预测求和；每组共有8541个有效全场目标对。

读取与导出示例（在本文件夹执行）：
 python tools/read_predictions.py experiments/comparison/seed_84/full/forecasts_h1_h6.npz
 python tools/read_predictions.py experiments/comparison/seed_84/full/forecasts_h1_h6.npz --csv seed84_full.csv --horizon 6
 python tools/read_predictions.py experiments/comparison/seed_84/full/forecasts_h1_h6.npz --csv seed84_full_all.csv
 CSV导出包含六步真实值、原评分标签、有效标记、预测、起报/目标时间；全部六步导出每组约数百MB。

校验：python tools/verify_package.py
从实际原始文件重建预处理/图缓存：python tools/rebuild_preprocessing.py --output rebuilt_cache
重建脚本不训练、不改变原时钟/划分；会输出X/Y、邻居、边、角色、时间等数组，约2GB。延迟对照的13点输入由已提供的features.npy和splits索引取得。
原模型训练入口：code/as_run/comparison_four_seeds/source_snapshot.py --data_csv <data/raw/sdwpf_2021_full.csv绝对路径> --location_csv <data/raw/sdwpf_turb_location_elevation.csv绝对路径> --seed 84 --experiments full --out_dir <新输出目录>。其他模型用run_index.json的case名。比较适配器主要复用缓存、保存checkpoint及分块写CSV；源码训练本体保持不变。
延迟对照实际运行与准备代码是code/as_run/delay_window/runner.py，计划参数在同目录plan.json。该冻结入口保留原机器路径；迁移运行需将plan中workspace/dataset_cache和输入路径指向本机重建位置，并在新计划中更新相应哈希，不能覆盖归档原计划。

统计口径：原脚本AVG和延迟主表均为六步合并（pooled）指标，不是逐步RMSE/R2算术平均。均值±SD是在每个种子得到指标后跨种子计算样本标准差。分层及仅有效标签重新评分见reports/observed_evaluation。不得把旧含填补标签评分与有效标签评分混列。
reports/figures包含两种明确分开的评分图。reports/efficiency的历史耗时是整组任务总墙钟耗时，包含初始化、预处理、训练、验证、测试和CSV导出，不是纯训练时间。所有已完成预测和数据处理均保留，未为本次打包重训。
manifest_sha256.json列出所有包内文件哈希。run_index.json保留预测来源和源文件哈希，可追溯到原工作区。ZIP旁的sha256文件用于核对上传包完整性。
'''

if __name__=='__main__':main()
