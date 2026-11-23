"""Read original run records; benchmark frozen checkpoints without retraining."""
import json
import hashlib
import importlib.util
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import winreg

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'计算效率_84_101_1234_20261122'
COHORT = ROOT/'对比实验_1234_21_42_84_20261119_164746'
NEW = ROOT/'对比实验_101_20261120_223936'
SEEDS = [84,101,1234]
WARMUP, ROUNDS, ITERATIONS = 30, 3, 100
LABELS = dict(full='Full',lstm='LSTM',cnn_lstm='CNN-LSTM',cnn_bilstm='CNN-BiLSTM',
              transformer='Transformer',dbfp_net='DBFP-Net',stgcn='STGCN',agcrn='AGCRN',
              stv_pgnn_adapted='STV-PGNN-adapted')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    plans = [read(x/'plan.json') for x in [COHORT,NEW]]
    assert plans[0]['source_sha256'] == plans[1]['source_sha256']
    for x,p in zip([COHORT,NEW],plans):
        assert read(x/'status.json')['status'] == 'complete'
        assert sha(x/'source_snapshot.py') == p['source_sha256']
    spec = importlib.util.spec_from_file_location('frozen_efficiency_source',COHORT/'source_snapshot.py')
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    cache = Path(plans[0]['dataset_cache'])
    arrays = {k:np.load(cache/f'test_{k}.npy',mmap_mode='r',allow_pickle=False)
              for k in ['X_time','nbr_idx','edge_attr','edge_mask','role']}
    cpu = winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
            r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'),'ProcessorNameString')[0].strip()
    gpu = torch.cuda.get_device_properties(0)
    hardware = dict(cpu=cpu,gpu=gpu.name,gpu_memory_GiB=gpu.total_memory/2**30,
                    ram_GiB=psutil.virtual_memory().total/2**30,os=platform.platform(),
                    python=sys.version,torch=torch.__version__,cuda_runtime=torch.version.cuda,
                    torch_threads=4,nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),
                    measured_at=__import__('datetime').datetime.now().isoformat())
    save(OUT/'hardware.json',hardware)
    all_runs, timings = [], []
    with torch.inference_mode():
        for mode in plans[0]['models']:
            for seed in SEEDS:
                folder = (NEW if seed == 101 else COHORT)/f'seed_{seed}'/mode
                done = read(folder/'completed.json')
                assert sha(folder/'summary_results.csv') == done['summary_sha256']
                history = pd.read_csv(folder/f'history_{mode}.csv')
                saved = torch.load(folder/'checkpoint_best.pt',map_location='cpu',weights_only=False)
                assert saved['seed'] == seed and saved['model'] == mode
                cfg = base.Config(**saved['config'])
                state = saved['model_state_dict']
                if mode == 'stv_pgnn_adapted':
                    cfg._stv_context = dict(xyz=state['xyz'].numpy(),mean=state['feature_mean'].numpy(),
                                            std=state['feature_std'].numpy())
                model = base.make_model(cfg,mode,6,n_turbines=134).cuda().eval()
                model.load_state_dict(state,strict=True)
                x = torch.from_numpy(arrays['X_time'][:1].copy()).float().cuda()
                physical = mode in {'full','stgcn'}
                batch = [x]
                for key in ['nbr_idx','edge_attr','edge_mask','role']:
                    a = arrays[key][:1].copy()
                    if not physical:
                        a.fill(0)
                    t = torch.from_numpy(a)
                    if key in {'edge_attr','role'}:
                        t = t.float()
                    batch.append(t.cuda())
                pred = model(*batch)
                assert pred.shape == (1,134,6) and torch.isfinite(pred).all()
                actual = pd.read_csv(folder/f'test_predictions_{mode}.csv',nrows=134*6)
                reference = actual.pred_kw.to_numpy().reshape(6,134).T
                predicted = pred[0].cpu().numpy()*cfg.rated_power_kw
                maxdiff = float(np.max(np.abs(predicted-reference)))
                # Historical evaluation used B=30. Convolution kernels can
                # change with B=1 (including TF32 paths), so verify checkpoints
                # at their original evaluation shape before benchmarking B=1.
                replay = [torch.from_numpy(arrays['X_time'][:30].copy()).float().cuda()]
                for key in ['nbr_idx','edge_attr','edge_mask','role']:
                    a = arrays[key][:30].copy()
                    if not physical:
                        a.fill(0)
                    t = torch.from_numpy(a)
                    replay.append((t.float() if key in {'edge_attr','role'} else t).cuda())
                replay_pred = model(*replay)[0].cpu().numpy()*cfg.rated_power_kw
                np.testing.assert_allclose(replay_pred,reference,rtol=1e-5,atol=.002)
                replay_diff = float(np.max(np.abs(replay_pred-reference)))
                del replay
                for _ in range(WARMUP):
                    model(*batch)
                torch.cuda.synchronize()
                wall, device = [], []
                start_event, end_event = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                for r in range(ROUNDS):
                    for i in range(ITERATIONS):
                        start_event.record()
                        t0 = time.perf_counter_ns()
                        model(*batch)
                        end_event.record()
                        end_event.synchronize()
                        ms = (time.perf_counter_ns()-t0)/1e6
                        dev = start_event.elapsed_time(end_event)
                        wall.append(ms); device.append(dev)
                        timings.append(dict(seed=seed,model=mode,round=r+1,iteration=i+1,
                                            synchronized_wall_ms=ms,cuda_event_ms=dev))
                row = dict(seed=seed,model=mode,label=LABELS[mode],epochs=len(history),
                           best_epoch=int(history.loc[history.val_node_avg_rmse.idxmin(),'epoch']),
                           task_elapsed_seconds=done['elapsed_seconds'],
                           inference_wall_mean_ms=float(np.mean(wall)),inference_wall_p50_ms=float(np.median(wall)),
                           inference_wall_p95_ms=float(np.percentile(wall,95)),
                           inference_device_mean_ms=float(np.mean(device)),
                           source_folder=str(folder),checkpoint_sha256=sha(folder/'checkpoint_best.pt'),
                           batch1_vs_historical_max_diff_kw=maxdiff,
                           checkpoint_batch30_replay_verified_max_diff_kw=replay_diff)
                assert row['epochs'] <= 50 and row['best_epoch'] == saved['best_epoch']
                all_runs.append(row)
                save(OUT/'runs.json',all_runs)
                print(f'{mode}/{seed}: epochs={len(history)}, task={done["elapsed_seconds"]:.1f}s, '
                      f'latency={np.mean(wall):.3f}ms',flush=True)
                del model,batch,pred
    save(OUT/'latency_trials.json',timings)
    summary = []
    for mode in plans[0]['models']:
        group = [r for r in all_runs if r['model'] == mode]
        row = dict(model=mode,label=LABELS[mode],seeds=SEEDS,
                   actual_epochs=[r['epochs'] for r in group],
                   task_seconds=[r['task_elapsed_seconds'] for r in group],
                   gpu='RTX 4060 Ti 8 GB')
        for output,key in [('task_seconds','task_elapsed_seconds'),('inference_ms','inference_wall_mean_ms')]:
            row[output+'_mean'] = float(np.mean([r[key] for r in group]))
            row[output+'_sd'] = float(np.std([r[key] for r in group],ddof=1))
        summary.append(row)
    save(OUT/'summary.json',summary)
    lines = ['模型 | 实际轮次(84/101/1234) | GPU | 原任务耗时(s,均值±SD) | 单次前向(ms,均值±SD)']
    for r in summary:
        lines.append(f'{r["label"]} | '+ '/'.join(map(str,r['actual_epochs'])) +
                     f' | {r["gpu"]} | {r["task_seconds_mean"]:.1f}±{r["task_seconds_sd"]:.1f}' +
                     f' | {r["inference_ms_mean"]:.3f}±{r["inference_ms_sd"]:.3f}')
    lines += ['',f'硬件（当前核查）：{cpu}；{gpu.name}，8 GB；内存约32 GB；Windows 10；PyTorch {torch.__version__}，CUDA runtime {torch.version.cuda}。',
              '训练耗时边界：历史elapsed_seconds是整组子进程墙钟耗时，包含进程初始化、数据处理/缓存载入、训练、验证、测试、预测CSV导出。纯训练时长未单独记录，不能将该数值改称纯训练耗时。',
              '历史采用两任务并行，第一批还与其他训练共享GPU；STV重新构建数据，其他模型复用缓存。因此原任务时长不能代表隔离条件下的算法计算效率。',
              '推理：27个最佳检查点，eval + inference_mode，float32，batch=1，L=3、N=134、H=6；每检查点预热30次，再分3轮各100次。',
              '表中延迟为CUDA同步墙钟前向耗时，包含CPU发起计算和GPU执行。输入已在GPU；排除文件读取、预处理、CPU图构建、CPU到GPU传输和结果导出。STV的网络内部物理图计算包含在内。',
              '这是一整个风场、全部6步的一次模型前向，不是单台风机或单预测步耗时，也不是部署端到端延迟。',
              '各检查点先求300次测量均值，再跨三个种子报告均值±样本标准差。原始延迟样本和每种子耗时保存在JSON中。',
              '已用每个检查点在历史评估batch=30下重放首个测试起报，并与历史预测CSV逐值核对，容差rtol=1e-5、atol=0.002 kW。batch=1可能选择不同卷积内核并产生数值差异，逐检查点差异另有记录。没有重新训练。']
    (OUT/'效率结果与计时口径.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'benchmark_protocol.json',dict(status='passed',runs=27,seeds=SEEDS,warmup=WARMUP,
          rounds=ROUNDS,iterations=ITERATIONS,batch=1,history=3,turbines=134,horizon=6,
          training_time_is_pure_training=False,runner_sha256=sha(Path(__file__))))
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
