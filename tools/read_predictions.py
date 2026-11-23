"""Inspect a curated six-horizon prediction archive; optionally export CSV."""
from pathlib import Path
import argparse,csv,numpy as np
p=argparse.ArgumentParser()
p.add_argument('file',type=Path)
p.add_argument('--csv',type=Path)
p.add_argument('--horizon',type=int,choices=range(1,7))
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
with np.load(a.file,allow_pickle=False) as z:pred=z['pred_kw']
with np.load(root/'data/shared_test_targets.npz',allow_pickle=False) as z:
    raw=z['true_kw_raw'];old=z['true_kw_as_run'];mask=z['observed_flag']
    origins=z['origin_time'];targets=z['target_time'];turbines=z['turbine_id']
print('pred_kw',pred.shape,'valid target pairs',int(mask.sum()))
if a.csv:
    a.csv.parent.mkdir(parents=True,exist_ok=True)
    with a.csv.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['origin_time','target_time','horizon','TurbID','true_kw_raw','observed_flag','true_kw_as_run','pred_kw'])
        for s in range(len(pred)):
            for h in ([a.horizon-1] if a.horizon else range(6)):
                for j,tid in enumerate(turbines):
                    w.writerow([str(origins[s]),str(targets[s,h]),h+1,int(tid),
                                float(raw[s,j,h]),bool(mask[s,j,h]),float(old[s,j,h]),float(pred[s,j,h])])
    print('CSV exported:',a.csv)
