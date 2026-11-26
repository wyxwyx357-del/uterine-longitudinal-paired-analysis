# Uterine Longitudinal Paired Analysis

用于分析**同一患者不同检查/不同周期**的超声变化，重点观察：

- 子宫内膜形态：厚度、容积、分型；
- 血流：VI、FI、VFI、PI、RI、S/D；
- 临床蠕动：正向/反向频率及总频率；
- 自动视频动态特征：前/后壁幅度为优先指标，R2、ER、MSE、M1 为探索性指标。

## 当前研究定位

这是一个小样本 pilot，优先利用“患者自己和自己比”的重复检查结构。

主问题不是建立妊娠预测模型，而是回答：

1. 同一患者两次检查时，内膜、血流和动态运动是否一起变化；
2. 对于两次妊娠结局不同的患者，成功周期与失败周期的指标变化方向是否一致；
3. 视频时长、检查时机、治疗方案、胚胎情况等是否可能解释观察到的差异。

## 主队列原则

主分析队列定义为：

- 同一患者至少 2 次可匹配检查；
- 每次分析视频可用时长 >= 60 s；
- 两次检查可明确配对；
- 缺失保持缺失，不自动填 0。

如果一个患者有超过 2 次检查，必须在本地数据中显式提供 `pair_order` 或预先指定配对规则；代码不自动挑“最有利”的两次检查。

## 当前优先指标

### 内膜
- thickness
- volume
- type（分类变量，单独描述）

### 血流
重点展示：
- VFI
- PI

同时保留：
- VI
- FI
- RI
- S/D

### 蠕动 / 动态
重点展示：
- 临床总蠕动频率 = 正向频率 + 反向频率
- anterior amplitude
- posterior amplitude

探索性：
- R2 frequency
- ER
- MSE
- M1 propagation/activity rate

## 分析顺序

1. 固定主队列与数据完整性；
2. 计算每位患者 `第二次 - 第一次` 的变化量；
3. 观察内膜、血流、蠕动变化是否同向；
4. 对结局不同的患者做“失败周期 vs 成功周期”自身对照；
5. 逐例核对治疗方案、检查时间、胚胎情况、视频时长和质量；
6. 只有在前述结果确实显示稳定联合变化后，才考虑综合表型。

## 不做的事情

当前 pilot 不以以下内容为目标：

- LASSO / SVM / XGBoost / 深度学习；
- 为提高 AUC 搜索特征或参数；
- 把 R2/M1 直接解释成真实生理波频率；
- 把 MSE 直接解释成已验证的双壁协调性生物标志物；
- 用缺失值自动写 0；
- 在公开仓库上传患者姓名、门诊号、原始临床表、视频、检查截图或任何可识别信息。

## 数据安全

**本仓库为公开仓库。严禁上传任何患者可识别信息。**

分析数据应在本地先完成去标识化，只保留研究用随机 ID。参见：

- `docs/DATA_DICTIONARY.md`
- `docs/STUDY_DESIGN.md`
- `docs/ANALYSIS_PLAN.md`

## 目录

```
docs/
  STUDY_DESIGN.md
  ANALYSIS_PLAN.md
  DATA_DICTIONARY.md
src/
  cohort_selection.py
  paired_change_analysis.py
  pregnancy_discordant_pairs.py
tests/
  test_paired_change_analysis.py
```

原始患者数据只保存在本地，不进入 Git。
