# Uterine longitudinal ultrasound project

本仓库现在明确拆成两个独立模块：

```text
原始超声视频
   ↓
01_feature_extraction/
   ↓  outputs/auto_features.csv
02_longitudinal_analysis/
   ↓
同一患者多次检查的纵向结果
```

## 01_feature_extraction

负责“视频怎么变成自动动态特征”：

- 追踪与运动校正；
- 径向运动与前/后壁幅度；
- ER / MSE；
- M1；
- 独立 R2 frequency；
- 技术QC；
- 最终导出一行一次检查的 `auto_features.csv`。

**当前该模块已经建立目录和输出协议，但原冻结追踪/特征源码尚未迁入。**

见 [01_feature_extraction/README.md](01_feature_extraction/README.md)。

## 02_longitudinal_analysis

负责“同一个患者不同检查怎么比较”：

- 每行一次检查的主表；
- strict / candidate 队列；
- 多次检查患者内轨迹；
- 相邻变化描述与QC；
- 形态—血流—动态患者内联动；
- 妊娠成功/失败周期自身对照。

该模块已有可运行的 1 个配置文件 + 4 个顺序脚本。

见 [02_longitudinal_analysis/README.md](02_longitudinal_analysis/README.md)。

## 安装纵向分析依赖

```bash
python -m pip install -r requirements.txt
```

## 隐私

这是公开仓库。不得提交患者姓名、门诊号、身份证号、电话、原始临床表、超声视频、DICOM、截图、患者级 NPZ/CSV 或带实名路径的信息。

真实数据和本地输出均由 `.gitignore` 屏蔽。
