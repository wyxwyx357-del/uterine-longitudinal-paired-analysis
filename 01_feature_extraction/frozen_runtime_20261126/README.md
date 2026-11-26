# Frozen current dynamic runtime — 2026-11-26

This directory is the formal integrated measurement runtime used by
`01_feature_extraction/run_feature_extraction.py`.

## Version

```text
UTERINE_DYNAMIC_FINAL_BATCH_20261126_V1
```

Current R2:

```text
R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106
```

M1:

```text
m1-global-pairing-unseen-v1
```

## Locks

- `SOURCE_LOCK.json`
- `FINAL_PIPELINE_SPEC.json`
- `r2_frozen/FROZEN_SPEC.json`

These define the current bundled workflow. They do not claim recovery of all
historical requested R2 SHA256 values.

## Outcome boundary

```text
outcomes_read = false
```

No pregnancy outcome is used by this measurement runtime.

## Output

`run_final_batch.py` generates:

```text
FINAL_DYNAMIC_FEATURES.csv
FINAL_RUN_SUMMARY.json
```

The repository-level wrapper publishes the exact CSV to:

```text
01_feature_extraction/outputs/FINAL_DYNAMIC_FEATURES.csv
```

## Environment

Use Python 3.10+ and the existing CUDA-enabled OpenCV build with
`cudaoptflow`.

Do not replace it with pip `opencv-python`.

Check before patient processing:

```bash
python verify_environment.py
python preflight_tests.py
```

## R2 inputs

R2 requires existing:

```text
video
r2_tracking
r2_nodes
mm_per_processed_pixel
```

The workflow does not synthesize missing R2 inputs.

## Missing values

Measurement failure remains missing. Zero is not used as a generic failure value.
