"""Verify curated GitHub package hashes and prediction arrays."""
from pathlib import Path
import hashlib,json,numpy as np
root=Path(__file__).resolve().parents[1]
for e in json.loads((root/'MANIFEST_SHA256.json').read_text(encoding='utf-8')):
    p=root/e['path'];assert p.stat().st_size==e['bytes'],p
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    assert h.hexdigest()==e['sha256'],p
runs=json.loads((root/'run_index.json').read_text(encoding='utf-8'));assert len(runs)==45
for r in runs:
    with np.load(root/r['prediction_file'],allow_pickle=False) as z:
        a=z['pred_kw'];assert a.shape==(7517,134,6) and np.isfinite(a).all(),r['prediction_file']
with np.load(root/'data/shared_test_targets.npz',allow_pickle=False) as z:
    assert z['true_kw_raw'].shape==(7517,134,6)
    assert int(z['observed_flag'].sum())==3753533
print('PASS: hashes, 45 prediction archives, shared labels')
