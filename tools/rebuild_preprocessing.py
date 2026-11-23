"""Rebuild the exact original preprocessing and physical graphs from bundled raw data."""
from pathlib import Path
import argparse,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert not a.output.exists(),'Choose a new output folder'
source=ROOT/'code/as_run/comparison_four_seeds/source_snapshot.py'
spec=importlib.util.spec_from_file_location('archived_preprocessing_source',source)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
plan=json.loads((ROOT/'code/as_run/comparison_four_seeds/plan.json').read_text(encoding='utf-8'))
cfg=m.Config(**plan['config']);cfg.data_csv=str(ROOT/'data/raw/sdwpf_2021_full.csv');cfg.location_csv=str(ROOT/'data/raw/sdwpf_turb_location_elevation.csv')
train,val,test,turbines,pred_times=m.build_dataset(cfg,'full');a.output.mkdir(parents=True)
sizes={}
for name,data in [('train',train),('val',val),('test',test)]:
    sizes[name]=len(data['X_time'])
    for key,value in data.items():np.save(a.output/f'{name}_{key}.npy',value,allow_pickle=False)
np.save(a.output/'pred_times.npy',pred_times.to_numpy())
assert sizes==dict(train=35075,val=7516,test=7517)
with np.load(ROOT/'data/shared_test_targets.npz') as targets:
    np.testing.assert_array_equal(test['times'],targets['target_time'])
    np.testing.assert_allclose(test['Y_matrix']*1500,targets['true_kw_as_run'],rtol=0,atol=1e-4)
(a.output/'ready.json').write_text(json.dumps(dict(samples=sizes,turbines=turbines,config=m.asdict(cfg)),ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: original preprocessing, graph construction, split sizes and test labels rebuilt')
