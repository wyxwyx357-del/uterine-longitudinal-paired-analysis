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

- `../PIPELINE_ORCHESTRATION_LOCK.json`
- `SOURCE_LOCK.json`
- `FINAL_PIPELINE_SPEC.json`
- `r2_frozen/FROZEN_SPEC.json`

Engineering revision:

```text
20261126_gemini_audit_fix3
```

This revision changes orchestration safeguards only; the locked measurement
definitions and thresholds are unchanged.

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


## Engineering safeguards

- case_id is preserved as a string across R2 CSV stages;
- relative manifest paths are resolved against the manifest directory;
- invalid/failed R2 cases are isolated and remain NaN with status/reason;
- merged R2 tables skip failed cases instead of indexing missing outputs;
- successful R2 cases continue and still produce FINAL_DYNAMIC_FEATURES.csv;
- temporary R2 cache files use a non-.npz suffix and cannot be discovered as completed caches;
- downstream longitudinal scripts preserve exam_id/patient_id as strings when re-reading master_long.csv;
- source hashes use canonical LF text hashing; .gitattributes also enforces LF.
