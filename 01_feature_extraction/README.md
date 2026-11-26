# 01 — Feature extraction

本模块负责：

```text
原始超声视频
→ 追踪 / 运动校正
→ 局部运动表征
→ 幅度 / ER / MSE / M1 / R2
→ QC
→ outputs/auto_features.csv
```

## 当前状态

**目前只建立目录、边界和输出协议。原冻结追踪/特征源码尚未迁入本仓库。**

后续迁移时原则是“迁移已有冻结实现”，而不是在这里重新发明算法。尤其：

- A0/P3 径向追踪链与 R2 频率链保持独立；
- 不因为纵向分析或妊娠结局修改追踪阈值、频率搜索或事件规则；
- 失败保持缺失，不自动写 0；
- 不跨无效追踪段拼接；
- 任何算法版本变化必须记录来源和版本。

## 目录

```text
01_feature_extraction/
├── tracking/          # A0/P3/LK、背景刚体校正、轨迹QC
├── radial_motion/     # A(s,t)、前后壁幅度
├── er_mse/            # ER、CC/MSE 等双壁表征
├── m1/                # 显著传播活动候选
├── r2_frequency/      # 独立 R2 局部主导频率链
├── qc/                # 技术QC、有效率、失败原因
├── OUTPUT_SCHEMA.md   # 与纵向分析的字段契约
└── outputs/           # 本地输出；Git忽略
```

## 唯一交接文件

完成特征提取后统一生成：

```text
01_feature_extraction/outputs/auto_features.csv
```

第二模块会默认从该路径读取，不需要人工复制。

字段定义见 [OUTPUT_SCHEMA.md](OUTPUT_SCHEMA.md)。
