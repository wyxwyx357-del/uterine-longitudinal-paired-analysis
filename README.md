# Uterine Longitudinal Paired Analysis

同一患者多次超声检查的纵向 pilot 分析：**内膜形态 + 血流 + 子宫动态运动**。

本仓库只保存代码、方法文档和不含患者信息的材料。真实临床表、视频、自动特征明细和分析输出全部保留在本地。

## 当前研究问题

1. 同一患者不同检查中，内膜形态、血流和动态运动如何变化；
2. 三类指标在患者内部是否存在稳定的联合变化关系；
3. 同时存在成功和失败周期的患者，成功/失败周期是否出现方向一致的患者内差异；
4. 观察到的变化是否可能由检查时机、治疗方案、胚胎情况、视频时长或追踪质量解释。

当前不做 LASSO、SVM、XGBoost、深度学习或综合评分，不以提高 AUC 为目标。

## 真实输入表

当前代码已经按现有 Excel 的 `临床数据` 工作表适配，直接识别：

- `检查ID`
- `患者分组`
- `患者内检查序号`
- `检查时间`
- `对应移植日期`
- `临床妊娠` / `生化妊娠`
- `子宫内膜厚度(mm)`
- `分型`
- `容积(ml)`
- `VI / FI / VFI`
- `子宫内膜血流参数｜S/D / PI / RI`
- `蠕动｜正向 / 蠕动｜反向`
- `治疗方式 / 治疗方案 / 周期类型`
- `胚胎情况 / 移植胚胎数`
- `匹配视频预计时长（秒）`

自动视频特征使用可选文件 `data_raw/auto_features.csv`，按 `检查ID/exam_id` 合并。支持的主要字段见 `config.py`。

## 队列定义

代码同时生成两个队列标签：

- **strict**：同一患者至少 2 次视频时长 >=60 s；正式 pilot 默认使用该组；
- **candidate**：同一患者至少 2 次有视频检查，并且至少 1 次 >=60 s；用于敏感性核对。

不会强行把患者压缩成两次检查。患者有 3、4 次或更多符合条件的检查时全部保留。

## 目录

```text
uterine-longitudinal-paired-analysis/
├── data_raw/                    # 本地真实数据；Git 忽略
├── outputs/
│   ├── tables/
│   └── figures/
├── config.py
├── 01_build_master_table.py
├── 02_delta_and_qc.py
├── 03_module_comovement.py
├── 04_outcome_self_control.py
├── docs/
└── requirements.txt
```

## 安装

```bash
python -m pip install -r requirements.txt
```

## 本地放数据

将临床 Excel 放到：

```text
data_raw/同一患者至少两次检查有视频_含时长.xlsx
```

如果已有自动视频特征，整理为一行一个检查的 CSV：

```text
data_raw/auto_features.csv
```

没有自动特征 CSV 也可以先运行；流水线会完成临床形态、血流和临床蠕动分析。

## 按顺序运行

```bash
python 01_build_master_table.py
python 02_delta_and_qc.py
python 03_module_comovement.py
python 04_outcome_self_control.py
```

### 01 — 主表与名单

输出核心文件：

- `master_long.csv`：每行一次检查；
- `cohort_summary.csv`：每位患者检查数、长视频数和队列标签；
- `candidate_not_strict.csv`：candidate 与 strict 的差异；
- `data_completeness.csv`：变量完整性。

患者源身份字段不会写入 `master_long.csv`，只生成本地伪匿名 `patient_id`。

### 02 — 相邻变化与 QC

保留所有纵向检查，计算：

[
\Delta_k = exam_k - exam_{k-1}
]

相邻 Δ 只用于个体轨迹、方向计数和描述性 QC。一个患者贡献多个区间时，不把这些区间当成多个独立患者做普通显著性检验。

同时检查自动动态特征的变化是否明显随视频时长或追踪有效率变化。

### 03 — 三模块患者内联动

预设少量有临床解释的组合，在原始长表上使用 repeated-measures correlation（`pingouin.rm_corr`），例如：

- 厚度 ↔ VFI
- 厚度 ↔ PI
- 容积 ↔ VFI
- VFI ↔ 临床总蠕动频率
- VFI ↔ 前/后壁幅度

不预设“全部一起升高”才算合理。结果必须结合个体轨迹、检查距移植天数和治疗方案解释。

### 04 — 成功/失败患者自身对照

筛选同一患者同时存在临床妊娠成功和失败周期者。

主分析：
- 每位患者先求成功周期中位值；
- 每位患者先求失败周期中位值；
- 计算患者级 `success median - failure median`；
- 报告方向人数、差值中位数/IQR；
- Wilcoxon P 值仅作描述。

敏感性分析：
- 每位患者预先按时间距离选择一对最近的不同结局检查；
- 再核对差异方向是否改变。

同时输出逐案核查 Excel，用于人工检查方案、检查距移植天数、胚胎情况、视频时长和 QC。

## 指标优先级

优先：
- 内膜厚度、容积；
- VFI、PI；
- 临床总蠕动频率；
- 前壁幅度、后壁幅度。

辅助：
- VI、FI、RI、S/D；
- 临床正向/反向频率和正向占比。

探索性：
- R2；
- 前/后壁 ER；
- MSE；
- M1。

R2/M1 不直接称为真实生理波频率；MSE 不直接称为已验证的双壁协调性生物标志物。

## 切换 candidate 敏感性分析

默认：

```python
ANALYSIS_COHORT = "strict"
```

位于 `config.py`。

strict 全部跑完后，将其改成：

```python
ANALYSIS_COHORT = "candidate"
```

再运行 02–04，即可得到 candidate 版本结果。01 不需要重新定义算法，只要主表不变即可。

## 隐私

这是公开仓库。严禁提交：

- 患者姓名、门诊号、身份证号、电话；
- 原始 Excel/CSV；
- 超声视频、DICOM、截图；
- 含患者姓名的本地路径；
- NPZ 等患者级追踪文件；
- 本地分析 outputs。

这些类型已在 `.gitignore` 中屏蔽。
