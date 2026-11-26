# Uterine longitudinal ultrasound project

本仓库用于同一患者多次超声检查的动态特征提取与纵向分析，正式拆成两个模块：

```text
视频 / 已有追踪与节点输入
        ↓
01_feature_extraction/
        ↓
outputs/FINAL_DYNAMIC_FEATURES.csv
        ↓
02_longitudinal_analysis/
        ↓
master_long.csv
        ↓
患者内变化 / 模块联动 / 成功失败自身对照
```

## 01_feature_extraction — measurement

负责“动态特征怎么算出来”。

当前正式 integrated runtime：

```text
01_feature_extraction/frozen_runtime_20261126/
```

正式 pipeline version：

```text
UTERINE_DYNAMIC_FINAL_BATCH_20261126_V1
```

包含当前锁定的：

- A0 tracking / 约2 s radial spacetime；
- anterior / posterior amplitude；
- ER；
- record MSE；
- M1 significant propagation activity；
- R2 current longitudinal posterior frequency；
- source/environment/preflight checks。

R2 正式版本：

```text
R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106
```

current R2 位于：

```text
01_feature_extraction/frozen_runtime_20261126/r2_frozen/
```

它是 current locked workflow，不声称等同于此前尝试恢复的 historical requested SHA set。

### Feature extraction run

manifest 模板：

```text
01_feature_extraction/frozen_runtime_20261126/manifest_template.csv
```

正式入口：

```bash
python 01_feature_extraction/run_feature_extraction.py \
  --manifest <manifest.csv> \
  --work-dir 01_feature_extraction/outputs/frozen_run \
  --r2-workers 1
```

成功后正式输出：

```text
01_feature_extraction/outputs/FINAL_DYNAMIC_FEATURES.csv
```

`auto_features.csv` 仅作为旧代码兼容投影，不是正式 source of truth。

### Environment

dynamic runtime 依赖：

```text
01_feature_extraction/frozen_runtime_20261126/requirements.txt
```

CUDA LK 必须使用已有 CUDA-enabled OpenCV + `cudaoptflow`。

不要安装普通 pip `opencv-python` 来替代 CUDA OpenCV，也不要静默退回 CPU 算法。

环境检查：

```bash
python 01_feature_extraction/frozen_runtime_20261126/verify_environment.py
python 01_feature_extraction/frozen_runtime_20261126/preflight_tests.py
```

详细来源：

- [Feature extraction README](01_feature_extraction/README.md)
- [Output schema](01_feature_extraction/OUTPUT_SCHEMA.md)
- [Source provenance](01_feature_extraction/SOURCE_PROVENANCE.md)

## 02_longitudinal_analysis — repeated examinations

负责“同一患者不同检查怎么分析”。

正式动态输入：

```text
01_feature_extraction/outputs/FINAL_DYNAMIC_FEATURES.csv
```

运行：

```bash
python 02_longitudinal_analysis/01_build_master_table.py
python 02_longitudinal_analysis/02_delta_and_qc.py
python 02_longitudinal_analysis/03_module_comovement.py
python 02_longitudinal_analysis/04_outcome_self_control.py
```

分析重点：

- 每行一次检查；
- 同一患者保留2/3/4+次合格检查；
- 相邻变化只作患者内轨迹/描述与QC；
- 预设少量形态—血流—动态 repeated-measures association；
- discordant-outcome患者做success vs failure自身对照；
- 当前pilot不做AUC hunting、LASSO/SVM/XGBoost/深度学习。

见 [Longitudinal analysis README](02_longitudinal_analysis/README.md)。

## Outcome blindness

动态 measurement chain 明确：

```text
outcomes_read = false
```

妊娠结局只进入第二模块统计阶段，不用于调整 tracking、frequency、M1、MSE 或 amplitude 算法。

## Missing data

统一原则：

```text
measurement failure != 0
```

R2/M1/MSE/amplitude/tracking 失败保持 blank / NaN，并尽可能保留 status / failure reason。

## Privacy

这是公开仓库。禁止提交：

- 姓名、门诊号、身份证号、电话；
- 原始临床表；
- 超声视频、DICOM、截图；
- 患者级 NPZ / tracking outputs；
- 实名本地路径；
- 本地 analysis outputs。

真实数据和本地输出由 `.gitignore` 屏蔽。
