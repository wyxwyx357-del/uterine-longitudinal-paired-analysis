# 02 — Longitudinal analysis

本模块负责：

```text
临床表
+
01_feature_extraction/outputs/FINAL_DYNAMIC_FEATURES.csv
        ↓
master_long.csv
        ↓
患者内纵向变化
        ↓
形态 / 血流 / 动态患者内联动
        ↓
成功 / 失败周期自身对照
```

## Formal dynamic-feature input

默认优先读取：

```text
../01_feature_extraction/outputs/FINAL_DYNAMIC_FEATURES.csv
```

这是 current frozen workflow 的正式输出。

只有在正式文件不存在时，才兼容读取历史投影：

```text
../01_feature_extraction/outputs/auto_features.csv
```

不要手工维护两份内容不同的特征表。

特征表中的：

```text
case_id
```

在加载层精确映射为：

```text
exam_id
```

必须唯一；不使用姓名模糊匹配。

## Clinical input

临床 Excel 默认：

```text
02_longitudinal_analysis/data_raw/同一患者至少两次检查有视频_含时长.xlsx
```

sheet：

```text
临床数据
```

如路径不同，只修改 `config.py`。

## Run order

在仓库根目录：

```bash
python 02_longitudinal_analysis/01_build_master_table.py
python 02_longitudinal_analysis/02_delta_and_qc.py
python 02_longitudinal_analysis/03_module_comovement.py
python 02_longitudinal_analysis/04_outcome_self_control.py
```

## Four analysis stages

1. `01_build_master_table.py`
   - 一行一次检查；
   - 合并 clinical + formal dynamic features；
   - 生成 patient_id、exam_order、days_to_transfer；
   - 生成 strict / candidate 队列；
   - 保留动态特征 status / failure / provenance。

2. `02_delta_and_qc.py`
   - 患者内相邻变化；
   - interval_n 与 patient_n 分开报告；
   - 检查动态变化与 video duration / tracking support 等技术QC变化；
   - 相邻区间不当作独立患者做普通推断。

3. `03_module_comovement.py`
   - 在原 long table 上做预设 repeated-measures correlation；
   - 形态—血流—动态少量预设关系；
   - 同时保留患者个体轨迹；
   - endometrium type 按转变分析，不做数值相减。

4. `04_outcome_self_control.py`
   - 仅在统计阶段读取 clinical pregnancy；
   - 同一患者 success median vs failure median；
   - 每位患者每个特征贡献一个患者级差；
   - 最近时间的不同结局检查对作为敏感性分析。

## Missing data

measurement failure != zero。

R2、M1、MSE、amplitude 或 tracking 失败均保持 NaN / blank，并保留状态字段；本pilot不做统一中位数填补。

## Scope

当前是患者内纵向/机制探索，不做 LASSO、SVM、XGBoost、深度学习或 AUC hunting。
