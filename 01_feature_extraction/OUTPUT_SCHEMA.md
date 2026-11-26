# Feature extraction output schema

## Formal cross-module contract

The formal output of the locked current workflow is:

```text
01_feature_extraction/outputs/FINAL_DYNAMIC_FEATURES.csv
```

It is produced by the unchanged frozen runtime:

```text
01_feature_extraction/frozen_runtime_20261126/run_final_batch.py
```

through the repository wrapper:

```text
01_feature_extraction/run_feature_extraction.py
```

The wrapper copies the exact frozen `FINAL_DYNAMIC_FEATURES.csv` to the stable
module output path. It does not recalculate, impute, tune or filter features.

`auto_features.csv` is now **compatibility only**. `export_auto_features.py`
may generate it as a projected/renamed subset for older code, but the formal
source of truth is `FINAL_DYNAMIC_FEATURES.csv`.

## Main fields used by longitudinal analysis

| canonical longitudinal field | source in FINAL_DYNAMIC_FEATURES.csv | meaning |
|---|---|---|
| exam_id | case_id | examination ID |
| anterior_amplitude | anterior_amplitude | anterior radial stretch activity amplitude |
| posterior_amplitude | posterior_amplitude | posterior radial stretch activity amplitude |
| r2_frequency_cpm | R2_frequency_cpm | current frozen posterior longitudinal R2 |
| anterior_er | anterior_ER_mean_paired | anterior ER summary over paired windows |
| posterior_er | posterior_ER_mean_paired | posterior ER summary over paired windows |
| mse | record_MSE_time_equal | time-equal record ER-MSE |
| m1_prop_rate_min_per_min | m1_prop_rate_min_per_min | lower-bound significant propagation activity rate |
| tracking_valid_fraction | raw_valid_fraction | A0 raw usable support fraction |

The longitudinal loader maps `case_id -> exam_id` exactly and requires one row
per examination. It does not use name-based fuzzy matching.

## QC / provenance retained when available

The master table also retains current-workflow status fields such as:

- pipeline_version
- a0_status
- anterior_amplitude_status / posterior_amplitude_status
- mse_status / mse_decomposition_status
- record_MSE_z and MSE decomposition terms
- m1_status / m1_failure_reason / m1_algorithm_version
- m1_prop_rate_max_per_min / count range / effective observation time
- R2_status / R2_failure_reason / R2_version
- R2_retained_posterior_region_count

## Missing-data rule

Measurement failure is never converted to zero.

```text
tracking failure != 0
R2 failure      != 0 cpm
M1 no support   != 0 activity/min
MSE unavailable != 0
amplitude fail  != 0
```

Missing or failed measurements remain blank / NaN with status and failure
reason where the frozen workflow provides them.

## Interpretation boundaries

- **R2** = local wall-tangent tissue-motion dominant frequency candidate.
- **M1** = significant propagation-activity candidate rate.
- **ER** = directional-energy representation.
- **MSE** = anterior/posterior ER coordination/difference candidate.
- **Amplitude** = radial stretch activity amplitude.

None of these automatically equals a confirmed physiological peristaltic-wave
count or externally validated clinical biomarker.

## Outcome blindness

The formal feature-extraction workflow is outcome blind. Its locked spec states:

```text
outcomes_read = false
```

Pregnancy outcomes enter only in the longitudinal/statistical module.
