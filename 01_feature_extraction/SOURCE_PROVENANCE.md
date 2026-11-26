# Feature extraction source provenance

本目录遵循一个原则：**迁移并锁定已有测量实现，不根据临床结局重写测量算法。**

## 1. Formal integrated runtime

当前正式动态特征运行时位于：

```text
01_feature_extraction/frozen_runtime_20261126/
```

当前工作流版本：

```text
UTERINE_DYNAMIC_FINAL_BATCH_20261126_V1
```

锁文件：

- `frozen_runtime_20261126/SOURCE_LOCK.json`
- `frozen_runtime_20261126/FINAL_PIPELINE_SPEC.json`

该 runtime 整合当前 A0 / amplitude / ER / record MSE / M1 / R2 生产链。
特征计算阶段明确：

```text
outcomes_read = false
```

## 2. P3 / radial tracking source

独立 P3 / radial tracking 源码来源：

- `wyxwyx357-del/uterine-tracking`
- 该仓库记录的原始迁移基线：
  `wyxwyx357-del/end@a9cf11b95293e395e0a90a1e3340393d782a2d5d`

保留位置：

- `tracking/peristalsis_pipeline/`
- `tracking/entrypoints/`
- `tracking/MIGRATION_MANIFEST_SOURCE.md`

这套独立追踪源码用于来源审计和既有 P3 链复现，不应在未验证的情况下与 integrated runtime 中的 A0/R2 tracking 互相替换。

## 3. A0 / amplitude / ER source

研究包来源：

- `wyxwyx357-del/uterine-wave-propagation-features`

可读源码保留在：

- `wave_features/uterine_wave/`

正式 integrated runtime 同时保留它锁定的原包路径：

- `frozen_runtime_20261126/analysis/mse_pregnancy_association/code/measurement/`
- `frozen_runtime_20261126/uterine_wave/`

主要包括 geometry、CUDA LK tracking、deformation、amplitude、spectrum、modes、patient、radial_events、quality 等。

## 4. Frozen M1 / record MSE

M1 正式版本：

```text
m1-global-pairing-unseen-v1
```

来源提交记录在 current `SOURCE_LOCK.json` 中。

仓库中保留：

- `m1/frozen_source/`
- `er_mse/frozen_source/`
- `frozen_m1_mse_bundle/`

正式生产运行优先使用 `frozen_runtime_20261126/` 中由 `SOURCE_LOCK.json` 锁定的依赖，避免人工拼接不同快照。

## 5. R2 current workflow

当前正式 R2 不再视为“源码缺失”。

正式 current workflow 位于：

```text
01_feature_extraction/frozen_runtime_20261126/r2_frozen/
```

版本：

```text
R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106
```

版本/源锁：

- `frozen_runtime_20261126/r2_frozen/FROZEN_SPEC.json`
- `frozen_runtime_20261126/SOURCE_LOCK.json`

当前主 R2 定义：

- local wall tangent，cervix → fundus；
- 0.5–5.0 cpm；
- frequency grid step 0.05 cpm；
- posterior wall only；
- high-confidence + independently validated stable-medium regions；
- 对可用 posterior cervical / middle / fundal region 的 `optimized_candidate_frequency_cpm` 做等权平均；
- 无可用 posterior region 时保持 missing，不写 0。

正式检查级字段：

```text
R2_frequency_cpm
```

### Historical SHA boundary

2026-11-26 曾尝试恢复更早的 historical requested SHA 集。current workflow 与其中部分文件并非同 SHA，例如 `run_local_motion_frequency.py`。

因此：

> current workflow 是当前锁定工作流，不声称恢复或逐字等同于 historical requested SHA set。

历史旧 SHA 仅作为审计记录保存，不是当前生产锁。当前生产锁以 `SOURCE_LOCK.json` 为准。

## 6. Formal output

current runtime 生成：

```text
FINAL_DYNAMIC_FEATURES.csv
```

仓库包装入口：

```text
01_feature_extraction/run_feature_extraction.py
```

会把 frozen runtime 的结果原样发布到：

```text
01_feature_extraction/outputs/FINAL_DYNAMIC_FEATURES.csv
```

`auto_features.csv` 仅为旧代码兼容投影，不再是正式 source of truth。

## 7. Data governance

不得向公开仓库提交患者姓名、医院ID、视频、DICOM、患者级 tracking/NPZ、临床原表或本地 outputs。
