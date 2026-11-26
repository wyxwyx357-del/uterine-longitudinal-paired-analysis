# Uterine longitudinal ultrasound project

本仓库分成两个独立模块：

```text
原始超声视频 / 冻结缓存
        ↓
01_feature_extraction/
        ↓  outputs/auto_features.csv
02_longitudinal_analysis/
        ↓
同一患者多次检查的纵向结果
```

## 01_feature_extraction — 视频到自动特征

已迁入真实冻结源码，不是接口占位。

正式运行版本：

```text
01_feature_extraction/frozen_runtime_20261126/
```

包含：

- A0 tracking / pose-corrected measurement；
- 约 2 s 径向时空表征；
- 前/后壁 amplitude；
- ER；
- record-level MSE 及分解；
- M1 significant propagation activity candidate；
- 独立 R2 longitudinal frequency；
- environment / preflight / source lock；
- final batch runner。

正式版本：

- pipeline: `UTERINE_DYNAMIC_FINAL_BATCH_20261126_V1`
- M1: `m1-global-pairing-unseen-v1`
- R2: `R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106`

R2 冻结目录已做逐文件 Git blob 校验：**22/22 与冻结源字节一致**。

### 运行

先使用正式模板准备 manifest：

```text
01_feature_extraction/frozen_runtime_20261126/manifest_template.csv
```

然后：

```bash
python 01_feature_extraction/run_feature_extraction.py \
  --manifest <manifest.csv> \
  --work-dir 01_feature_extraction/outputs/frozen_run \
  --r2-workers 6 \
  --resume
```

成功后生成：

```text
01_feature_extraction/outputs/auto_features.csv
```

注意：A0 正式流程仍按原冻结实现需要 pose/校正输入。仓库没有为了“全自动”另写一套 pose 算法。

更多信息：

- [Feature extraction README](01_feature_extraction/README.md)
- [Output schema](01_feature_extraction/OUTPUT_SCHEMA.md)
- [Source provenance](01_feature_extraction/SOURCE_PROVENANCE.md)

## 02_longitudinal_analysis — 同患者多次检查分析

负责：

- 每行一次检查的长格式主表；
- strict / candidate 队列；
- 多次检查患者内轨迹；
- 相邻变化描述与技术 QC；
- 形态—血流—动态患者内联动；
- 成功/失败周期自身对照。

运行：

```bash
python 02_longitudinal_analysis/01_build_master_table.py
python 02_longitudinal_analysis/02_delta_and_qc.py
python 02_longitudinal_analysis/03_module_comovement.py
python 02_longitudinal_analysis/04_outcome_self_control.py
```

见 [02_longitudinal_analysis/README.md](02_longitudinal_analysis/README.md)。

## 环境

纵向分析：

```bash
python -m pip install -r requirements.txt
```

正式特征提取：

```bash
python -m pip install -r 01_feature_extraction/frozen_runtime_20261126/requirements.txt
```

**不要用 pip 的 `opencv-python` 覆盖项目原有 CUDA-enabled OpenCV。**
追踪依赖 CUDA `cudaoptflow`；不满足环境时停止，不静默退回另一套算法。

## 隐私

这是公开仓库。禁止提交：

- 患者姓名、门诊号、身份证号、电话；
- 原始临床 Excel / CSV；
- 超声视频、DICOM、截图；
- 患者级 NPZ / NPY / 追踪输出；
- 含实名信息的本地路径；
- 本地分析 outputs。

真实数据与生成输出已由 `.gitignore` 屏蔽。
