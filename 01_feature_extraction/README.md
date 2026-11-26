# 01 — Feature extraction

本模块负责：

```text
原始超声视频
→ 追踪 / 运动校正
→ 局部运动表征
→ 幅度 / ER / MSE / M1 / R2
→ QC
→ 自动特征表
```

## 当前迁移状态

### 已迁入

1. **P3 / radial tracking**
   - 来源：`wyxwyx357-del/uterine-tracking`
   - 路径：`tracking/peristalsis_pipeline/`
   - 原追踪入口：`tracking/entrypoints/`
   - 来源迁移清单：`tracking/MIGRATION_MANIFEST_SOURCE.md`

2. **A0 / radial deformation / amplitude / ER / mode features**
   - 来源：`wyxwyx357-del/uterine-wave-propagation-features`
   - 路径：`wave_features/uterine_wave/`
   - 冻结入口：`wave_features/run_frozen_features.py`

3. **Frozen M1 / record MSE**
   - 来源分支：`m1-global-pairing-unseen-v1-source-20261126`
   - 功能副本：`m1/frozen_source/`、`er_mse/frozen_source/`
   - 原包结构：`frozen_m1_mse_bundle/`

### 尚未迁入

4. **R2 local dominant frequency**
   - 当前只恢复到冻结源码文件名与 SHA256；
   - Python 源文件本体当前不可访问，因此没有根据报告重写；
   - 校验清单：`r2_frequency/EXPECTED_SOURCE_HASHES.json`

详细来源见 [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md)。

## 运行 wave feature 包

```bash
cd 01_feature_extraction/wave_features
python -m pip install -e .
```

CUDA LK 需要带 `cudaoptflow` 的 CUDA OpenCV。普通 `opencv-python` wheel 不等价于原 CUDA 环境，不能静默回退 CPU。

冻结特征入口：

```bash
python run_frozen_features.py audit ...
python run_frozen_features.py run ...
```

具体参数沿用源码入口和原仓库说明。

## P3 tracking

P3 追踪保持原独立实现。源码在：

```text
tracking/peristalsis_pipeline/
```

完整追踪入口在：

```text
tracking/entrypoints/run_tracking_only_pipeline.py
```

该分支和 wave_features 中的 A0 tracking 不是“同一个文件换名字”，不得在不验证的情况下互相替换。

## 输出到纵向分析

纵向模块的目标输入仍为：

```text
01_feature_extraction/outputs/auto_features.csv
```

目前各冻结分支已经能产生各自原始结果，但**统一 auto_features.csv 的最终聚合器尚未冻结**。在 R2 源码到位前，不应假装已经形成“所有动态特征一键导出”的最终生产链。

## 数据边界

- 不读取妊娠结局来调整追踪/特征算法；
- 缺失保持缺失，不补 0；
- 不跨失追、reanchor、segment/pose epoch 断点拼接；
- R2、M1、ER/MSE 各自保持原定义边界；
- 患者数据和本地输出不提交 Git。
