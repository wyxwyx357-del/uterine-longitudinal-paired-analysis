# Uterine longitudinal ultrasound project

本仓库拆成两个独立模块：

```text
原始超声视频
   ↓
01_feature_extraction/
   ↓  dynamic feature outputs
02_longitudinal_analysis/
   ↓
同一患者多次检查的纵向结果
```

## 01_feature_extraction

负责“视频怎么变成自动动态特征”。

当前已迁入：

- P3 / radial tracking 源码；
- A0 / radial deformation；
- 前后壁 amplitude；
- ER / mode-window coordination；
- frozen record MSE；
- frozen M1 significant propagation activity；
- 对应运行入口和来源记录。

R2 local dominant frequency 仍保持独立分支。目前只恢复到冻结源码文件名与 SHA256，源码 Python 文件本体尚未在可访问来源中找到，因此没有根据报告重写。

详细状态：

- [01_feature_extraction/README.md](01_feature_extraction/README.md)
- [01_feature_extraction/SOURCE_PROVENANCE.md](01_feature_extraction/SOURCE_PROVENANCE.md)

## 02_longitudinal_analysis

负责“同一个患者不同检查怎么比较”。

包括：

- 每行一次检查的长格式主表；
- strict / candidate 队列；
- 多次检查患者内轨迹；
- 相邻变化描述与 QC；
- 形态—血流—动态患者内联动；
- 妊娠成功/失败周期自身对照。

运行入口：

```bash
python 02_longitudinal_analysis/01_build_master_table.py
python 02_longitudinal_analysis/02_delta_and_qc.py
python 02_longitudinal_analysis/03_module_comovement.py
python 02_longitudinal_analysis/04_outcome_self_control.py
```

见 [02_longitudinal_analysis/README.md](02_longitudinal_analysis/README.md)。

## 当前模块交接

纵向模块预期统一自动特征文件：

```text
01_feature_extraction/outputs/auto_features.csv
```

目前底层冻结算法已迁入，但“所有分支统一聚合为 auto_features.csv”的最终生产聚合器尚未冻结，尤其 R2 源码尚缺，因此不会伪装成已完成的一键全特征流水线。

## 安装纵向分析依赖

```bash
python -m pip install -r requirements.txt
```

wave feature 包：

```bash
cd 01_feature_extraction/wave_features
python -m pip install -e .
```

CUDA LK 仍要求带 cudaoptflow 的 CUDA OpenCV；普通 opencv-python 不等价于原运行环境。

## 隐私

这是公开仓库。不得提交：

- 患者姓名、门诊号、身份证号、电话；
- 原始临床表；
- 超声视频、DICOM、截图；
- 患者级 NPZ/CSV；
- 带实名路径的信息；
- 本地分析 outputs。

真实数据和本地输出均由 `.gitignore` 屏蔽。
