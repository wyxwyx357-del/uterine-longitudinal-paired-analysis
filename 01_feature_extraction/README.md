# 01 — Feature extraction

本模块负责：

```text
视频 / 既有 tracking / nodes
→ tracking / motion correction
→ local motion representation
→ amplitude / ER / MSE / M1 / R2
→ QC / status / provenance
→ FINAL_DYNAMIC_FEATURES.csv
```

## Formal production runtime

当前正式 integrated runtime：

```text
frozen_runtime_20261126/
```

版本：

```text
UTERINE_DYNAMIC_FINAL_BATCH_20261126_V1
```

锁文件：

```text
frozen_runtime_20261126/SOURCE_LOCK.json
frozen_runtime_20261126/FINAL_PIPELINE_SPEC.json
```

该 runtime 是当前工作流的正式执行来源。不要把 `wave_features/`、`m1/frozen_source/` 或独立 P3 仓库中的文件人工拼成另一条生产主链。

## Current R2 is present

R2 current workflow 已完整存在：

```text
frozen_runtime_20261126/r2_frozen/
```

版本：

```text
R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106
```

正式 exam-level R2：

```text
retained posterior cervical/middle/fundal optimized frequency
→ equal mean
→ R2_frequency_cpm
```

无 retained posterior region 时保持 NaN，并输出 R2 status/failure reason。

历史 requested SHA 不是 current production lock；详见：

```text
r2_frequency/HISTORICAL_REQUESTED_SOURCE_HASHES.json
```

current production lock 始终以：

```text
frozen_runtime_20261126/SOURCE_LOCK.json
```

为准。

## Formal run

manifest 模板：

```text
frozen_runtime_20261126/manifest_template.csv
```

入口：

```bash
python 01_feature_extraction/run_feature_extraction.py \
  --manifest <manifest.csv> \
  --work-dir 01_feature_extraction/outputs/frozen_run \
  --r2-workers 1
```

正式输出：

```text
outputs/FINAL_DYNAMIC_FEATURES.csv
```

兼容旧代码时才额外生成：

```text
outputs/auto_features.csv
```

可以使用 `--no-compat-auto-features` 关闭兼容投影。

## Input expectations

combined current workflow 的 manifest 可以包含：

```text
case_id
video
annotation
pose
time_factor
a0_spacetime_cache
a0_tracks_cache
a0_layout_cache
r2_tracking
r2_nodes
mm_per_processed_pixel
```

A0 若已有完整 spacetime cache 可复用；否则需要 video + annotation + pose。

R2 需要现有 video + R2 tracking seed + nodes + pixel scale。当前工作流不会自动生成缺失的 R2 输入。

## Main feature fields

- anterior_amplitude
- posterior_amplitude
- anterior_ER_mean_paired
- posterior_ER_mean_paired
- record_MSE_time_equal
- m1_prop_rate_min_per_min
- R2_frequency_cpm
- raw_valid_fraction

同时保留 status、failure reason、MSE decomposition、M1 count range、R2 retained region count 等。

字段契约见 [OUTPUT_SCHEMA.md](OUTPUT_SCHEMA.md)。

## Independent source archives

为可追溯性，仓库仍保留：

- `tracking/`：独立 P3/radial tracking 迁移源码；
- `wave_features/`：A0/radial wave research package；
- `m1/frozen_source/`、`er_mse/frozen_source/`：功能分类副本；
- `frozen_m1_mse_bundle/`：原 import 路径结构。

这些用于审计/复现来源，不取代 formal integrated runtime。

## Environment

```bash
python -m pip install -r 01_feature_extraction/frozen_runtime_20261126/requirements.txt
```

requirements 故意不安装普通 `opencv-python`。

必须保留 CUDA-enabled OpenCV with `cudaoptflow`。环境不满足时停止，不能换算法。

## Outcome blindness and missing data

current spec：

```text
outcomes_read = false
missing_policy = leave blank/NaN; never replace with zero
```

动态特征阶段不读取妊娠结局，也不根据AUC改阈值。
