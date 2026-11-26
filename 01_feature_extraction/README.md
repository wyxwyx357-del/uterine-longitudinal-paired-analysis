# 01 — Feature extraction

本模块负责完整的：

```text
原始超声视频 / 已有冻结缓存
→ A0 追踪与运动校正
→ 径向时空表征
→ 幅度 / ER / MSE / M1
→ 独立 R2 frequency
→ 技术 QC
→ outputs/auto_features.csv
```

## 正式运行版本

正式入口位于：

```text
frozen_runtime_20261126/
```

核心版本：

- pipeline: `UTERINE_DYNAMIC_FINAL_BATCH_20261126_V1`
- M1: `m1-global-pairing-unseen-v1`
- R2: `R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106`

该目录是当前正式运行源。算法源码来自既有冻结代码，不因本纵向研究重新调参。

## 一键运行

准备 manifest 后：

```bash
python 01_feature_extraction/run_feature_extraction.py \
  --manifest <manifest.csv> \
  --work-dir 01_feature_extraction/outputs/frozen_run \
  --r2-workers 6 \
  --resume
```

成功后自动产生：

```text
01_feature_extraction/outputs/auto_features.csv
```

纵向分析模块默认直接读取这个文件。

## Manifest

正式 manifest 模板：

```text
frozen_runtime_20261126/manifest_template.csv
```

字段包括：

- `case_id`
- `video`
- `annotation`
- `pose`
- `time_factor`
- 可选 A0 已有缓存
- `r2_tracking`
- `r2_nodes`
- `mm_per_processed_pixel`

A0 正式流程仍按原冻结设计使用 pose/校正输入。本次迁移**没有新增或替换 pose 算法**。

## 代码分层

### `frozen_runtime_20261126/`

当前真正运行的冻结全流程。包含：

- A0 measurement dependencies
- amplitude / ER / MSE
- M1
- R2
- final batch runner
- environment/preflight checks
- source lock

### `a0_wave_pipeline/`

历史 `3b6e054...` A0/波特征审计快照，保留用于追溯和对照。

### `m1/frozen_source/`

M1 独立冻结来源快照，来源提交 `75d26aca...`。

### `er_mse/frozen_source/`

MSE patient-level 核心实现来源快照。

## 环境

正式运行依赖：

```bash
python -m pip install -r 01_feature_extraction/frozen_runtime_20261126/requirements.txt
```

**不要安装 pip 的 `opencv-python` 去替换现有环境。**

追踪需要原项目使用的 CUDA-enabled OpenCV / `cudaoptflow`。环境不满足时应停止，不自动退回 CPU 算法。

## 输出边界

统一输出字段见 [OUTPUT_SCHEMA.md](OUTPUT_SCHEMA.md)。

- 特征失败保持缺失；
- 不用 0 代替失败；
- 不读取妊娠结局；
- 不因纵向结果修改追踪/特征定义；
- R2、M1、MSE 保持各自原始解释边界。

来源与迁移校验见 [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)。
