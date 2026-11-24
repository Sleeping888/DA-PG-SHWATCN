# DA-PG-SHWATCN reproducibility package

Training code, run configurations, checkpoints, training records, split indices and six-horizon
predictions for the reported experiments: 27 comparison runs (seeds 84, 101 and 1234 across nine
models) and 18 delay/history runs. `run_index.json` indexes every run, `MANIFEST_SHA256.json` holds
the upload-file hashes and `release_info.json` records the release metadata. Local paths in text
files and source snapshots are replaced by `project_root`; see `PATH_REDACTION.md`.

## Predictions

`predictions/<study>/<seed>/<case>.npz` holds one `pred_kw` float32 array of shape [7517, 134, 6] in
kW, ordered by test origin, turbine ID and horizon 1-6. `data/shared_test_targets.npz` holds
`true_kw_raw`, `true_kw_as_run`, `observed_flag`, `origin_time`, `target_time` and `turbine_id`.
Scoring on observed labels uses `observed_flag` with `true_kw_raw`; scoring on the filled series uses
`true_kw_as_run`. Farm-level observed-label scoring requires all 134 labels at an origin to be valid.

```python
import numpy as np
pred = np.load('predictions/comparison/84/full.npz', allow_pickle=False)['pred_kw']
with np.load('data/shared_test_targets.npz', allow_pickle=False) as z:
    raw = z['true_kw_raw']; valid = z['observed_flag']; times = z['target_time']
print(pred.shape, raw.shape, valid.sum())
```

## Tools

`tools/verify_package.py` checks the manifest hashes and the shape of every prediction archive.
`tools/read_predictions.py` prints a run or exports one horizon to CSV.

## Data and split

The raw SDWPF files and the feature caches are not included. Obtain the 2021 records from
https://doi.org/10.6084/m9.figshare.24798654 and rebuild with `code/as_run` and `tools`. The split is
chronological and not purged, so five target timestamps overlap at each boundary
(`splits/definitions.json`); the retained time axis contains 59 intervals that are not 10 minutes.

## Runs and variants

`code/as_run/<study>/runner.py` reproduces a study from its `plan.json`. Two variants that the code
supports are not reported in the paper: `no_hetero`, a physical wake graph without soft-role gating
and without the adaptive branch, and `mtgnn`. In the paper, `w/o Delay` and `w/o Farm Loss` are the
full model run with `--use_wake_delay 0` and with `--farm_loss_weight 0`.

Run-level outputs for the ablation variants of Table 6 and for the coefficient scan of Table 9 are
not included.
