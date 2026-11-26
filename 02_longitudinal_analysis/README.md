# 02 — Longitudinal analysis

本模块负责：

```text
临床表 + 01_feature_extraction/outputs/auto_features.csv
→ 长格式主表
→ 患者内纵向变化
→ 形态/血流/动态患者内联动
→ 成功/失败周期自身对照
```

## 本地输入

临床 Excel 放到：

```text
02_longitudinal_analysis/data_raw/同一患者至少两次检查有视频_含时长.xlsx
```

自动特征默认读取：

```text
../01_feature_extraction/outputs/auto_features.csv
```

如果自动特征文件暂时不存在，01 脚本仍可先建立临床形态、血流和临床蠕动主表。

## 运行

在仓库根目录：

```bash
python 02_longitudinal_analysis/01_build_master_table.py
python 02_longitudinal_analysis/02_delta_and_qc.py
python 02_longitudinal_analysis/03_module_comovement.py
python 02_longitudinal_analysis/04_outcome_self_control.py
```

也可以先进入本目录再运行。

## 四层分析

1. `01_build_master_table.py`：拼出每行一次检查的长表、派生临床蠕动指标、生成 strict/candidate 队列。
2. `02_delta_and_qc.py`：患者内相邻变化、方向统计、轨迹及技术QC；相邻区间不当作独立患者做普通推断。
3. `03_module_comovement.py`：直接在长表上做预设组合的 repeated-measures correlation，并保留个体轨迹和时间/方案背景。
4. `04_outcome_self_control.py`：同一患者成功/失败周期汇总比较、方向人数、探索性 Wilcoxon、最近不同结局检查敏感性分析。

详细设计见 [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md)。
