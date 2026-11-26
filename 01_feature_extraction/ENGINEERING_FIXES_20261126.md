# Engineering hardening after Gemini final audit — 2026-11-26

Revision:

```text
20261126_gemini_audit_fix2
```

Fix2 additionally closes the two MINOR findings from the first post-fix re-audit:
- `export_auto_features.py` validates the `case_id` column and missing values before string coercion, so malformed NaN IDs cannot become the literal string `"nan"`;
- `preflight_tests.py` now uses `Path(__file__).resolve().parent` as the frozen-runtime import root.

This revision is intentionally limited to orchestration, identity preservation,
input validation and reproducibility safeguards. It does **not** change the
locked mathematical definitions or thresholds of A0, amplitude, ER, MSE, M1 or
R2.

## Fixed BLOCKER 1 — source-lock verification

- corrected `verify_environment.py` so `SOURCE_LOCK.json` paths are resolved
  relative to `frozen_runtime_20261126/`, where the locked source actually lives;
- added repository `.gitattributes` with LF line endings for text source files;
- hash verification canonicalizes text line endings to LF, preventing Windows
  `core.autocrlf` from producing false source-lock failures;
- added `01_feature_extraction/PIPELINE_ORCHESTRATION_LOCK.json` for the formal
  entry/wrapper/spec files that were previously outside `SOURCE_LOCK.json`.

## Fixed BLOCKER 2 — one R2 case must not abort the cohort

- `r2_frozen/run_tracking_batch.py` now merges all successfully completed cases
  even when other cases fail;
- failed cases are recorded in `merged_results/failed_cases.csv` and in
  `resumable_status.json`;
- downstream cache discovery ignores caches from cases whose tracking state is
  not `complete`;
- `run_final_batch.py` carries failed/invalid R2 cases forward as
  `R2_frequency_cpm = NaN` plus `R2_status` and `R2_failure_reason`;
- an unexpected whole-stage R2 exception is isolated from A0/M1/MSE so the final
  examination table can still be produced with explicit R2 failure status.

No failed R2 measurement is replaced by zero.

## Fixed BLOCKER 3 — case/exam ID type stability

- manifest `case_id` remains a string from CSV input onward;
- R2 pandas CSV readers explicitly preserve `case_id` as string;
- compatibility export preserves `case_id` before renaming to `exam_id`;
- longitudinal clinical/feature merge preserves exam identifiers as strings;
- dynamic feature rows whose exam IDs do not exist in the clinical table now
  raise an explicit error instead of being silently dropped.

This protects identifiers such as `00123` from accidental conversion to
`123`.

## Fixed MAJOR — relative manifest paths

Formal manifest path fields are resolved relative to the manifest file's own
directory before A0/R2 processing:

- video
- annotation
- pose
- a0_spacetime_cache
- r2_tracking
- r2_nodes

R2 therefore receives absolute paths even when the user supplies portable
relative manifest paths.

## Explicit geometry preflight

The current R2 algorithm still has its existing source-video geometry
requirement:

```text
1260 x 910
```

This algorithmic assumption was **not changed**. Instead,
`verify_tracking_inputs.py` now reports
`unsupported_video_geometry:<width>x<height>;expected_1260x910` before
tracking. The incompatible case is isolated rather than allowed to crash the
cohort.

Changing the R2 background ROI or geometry normalization would be a measurement
algorithm change and is outside this engineering revision.

## Formal manifest cleanup

The formal manifest template no longer advertises
`a0_tracks_cache` / `a0_layout_cache`, because the formal
`run_final_batch.py` does not consume them.

The deprecated broken `run_final_batch_full.py` wrapper was removed from the
formal runtime. Git history retains it for audit.

## Lock state after fixes

- orchestration lock: verified against current Git blob SHAs;
- modified R2 source files: verified against current canonical-LF SHA256 values
  in `SOURCE_LOCK.json`;
- `algorithm_definition_unchanged = true` is recorded in the lock/spec.

## Still intentionally unchanged

The following were **not** modified:

- A0 tracking parameters;
- amplitude definition;
- ER 20 s / 0.5–5 cpm directional spectrum definition;
- record MSE formulas;
- M1 thresholds/identity/count definition;
- R2 0.5–5.0 cpm grid and 0.05 cpm step;
- R2 fixed-point/continuous-segment rules;
- R2 high + validated-stable-medium retention;
- final posterior retained-region equal-mean aggregation;
- R2 hard-coded background ROI / source geometry algorithm.

## Required local validation before full patient batch

On the target CUDA machine, run:

```bash
python 01_feature_extraction/frozen_runtime_20261126/verify_environment.py
python 01_feature_extraction/frozen_runtime_20261126/preflight_tests.py
```

Then perform a small engineering pilot containing at least:

1. one normal case;
2. one deliberately invalid/failed R2 case;
3. one string ID with leading zero, e.g. `00123`;
4. one manifest using relative paths.

Expected behavior: the valid cases continue, failed measurements remain NaN
with status/reason, IDs remain exact strings, and
`FINAL_DYNAMIC_FEATURES.csv` is still produced.
