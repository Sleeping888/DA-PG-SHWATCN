# DA-PG-SHWATCN reproducibility package

This curated package includes the manuscript comparison seeds 84, 101 and 1234 (27 model runs), 18 delay/history runs, exact run configurations, checkpoints, training records, split indices, source snapshots, and their complete six-horizon predictions. Each run is indexed in `run_index.json`. Local paths in text files are redacted for double-anonymized review; see `PATH_REDACTION.md`.

Each file `predictions/<study>/<seed>/<case>.npz` contains a single `pred_kw` float32 array of shape [7517,134,6] in kW: test origin, turbine ID order, horizons 1–6. `data/shared_test_targets.npz` contains `true_kw_raw`, `true_kw_as_run`, `observed_flag`, `origin_time`, `target_time` and `turbine_id`. The original-valid scoring uses `observed_flag` and `true_kw_raw`; the original filled-label scoring uses `true_kw_as_run`. Farm observed-label scoring requires all 134 labels valid.

Example:
```python
import numpy as np
pred = np.load('predictions/comparison/84/full.npz', allow_pickle=False)['pred_kw']
with np.load('data/shared_test_targets.npz', allow_pickle=False) as z:
    raw = z['true_kw_raw']; valid = z['observed_flag']; times = z['target_time']
print(pred.shape, raw.shape, valid.sum())
```

`tools/verify_package.py` checks file hashes and all prediction shapes. `tools/read_predictions.py` can inspect a run or export one horizon to CSV. The original six-horizon archive SHA-256 for each run is retained in `run_index.json`; the new upload-file hashes are in `MANIFEST_SHA256.json`.

Raw SDWPF data and generated feature caches are excluded. Obtain the source data from https://doi.org/10.6084/m9.figshare.24798654 and rebuild with `code/as_run` and `tools`. `splits/definitions.json` records five overlapping target timestamps at each chronological split boundary; the split is not purged. The retained time axis has 59 non-10-minute gaps.

**Not in the supplied archive:** run-level files for the paper's ablation variants (Table 6) and wake-coefficient sensitivity scan (Table 9). Those studies cannot be independently reproduced from this folder. The GitHub repository is https://github.com/Sleeping888/DA-PG-SHWATCN; verify its contents and create an immutable tag before referring to the package as an archived paper release.
