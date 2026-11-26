# Pilot analysis plan

本项目按 4 层顺序执行，默认主分析为 strict 队列。

## 1. 建立长格式主表

`01_build_master_table.py`

- 每行 = 一次检查；
- 同一患者所有合格检查全部保留；
- clinical Excel 已按现有 `临床数据` 工作表适配；
- 自动视频特征按 `exam_id/检查ID` 合并；
- 生成 strict 与 candidate 队列标签；
- 源姓名/医院身份字段不写入分析主表。

strict：每位患者至少 2 次 >=60 s 检查。

candidate：每位患者至少 2 次有视频检查，且至少 1 次 >=60 s。

## 2. 纵向变化与技术 QC

`02_delta_and_qc.py`

按患者时间顺序计算所有相邻变化：

`delta_k = exam_k - exam_(k-1)`

输出：
- 相邻区间数；
- 涉及患者数；
- delta 中位数与 IQR；
- 上升/下降/不变区间数；
- 每位患者的 median adjacent delta 方向；
- 个体轨迹图；
- 动态指标 delta 与视频时长/追踪有效率 delta 的描述性 Spearman rho。

重要：同一患者的多个相邻 delta 不独立，因此不把这些区间拼池后做普通显著性检验。

## 3. 形态—血流—蠕动患者内联动

`03_module_comovement.py`

直接在每行一次检查的长表上做 repeated-measures correlation（rmcorr），不把相邻 delta 当独立样本。

预设关键组合写在 `config.py::KEY_PAIRS`。

结果必须同时看：
- rmcorr；
- 每位患者轨迹；
- days_to_transfer；
- 治疗方案是否变化。

另外描述内膜分型发生变化 vs 未变化的相邻区间中，血流/幅度/临床蠕动的变化。

不预设形态、血流、蠕动必须同向增加。

## 4. 妊娠结局不同患者的自身对照

`04_outcome_self_control.py`

纳入同一患者同时有成功和失败周期者。

主分析：
- 每患者成功周期特征中位数；
- 每患者失败周期特征中位数；
- success median - failure median；
- 患者方向人数；
- 差值中位数/IQR；
- Wilcoxon P 值仅作探索性描述。

敏感性分析：
- 每位患者只保留时间距离最近的一对不同结局检查；
- 比较方向是否与全部周期汇总一致。

逐案核查表并排保留：
- 检查距移植时间；
- 治疗方式/方案；
- 周期类型；
- 胚胎情况/移植数；
- 视频时长；
- 追踪有效率/QC/pose 来源（如有）。

## 当前禁区

不做：
- LASSO/SVM/XGBoost/深度学习；
- 为提高 AUC 搜索特征；
- 小样本综合评分；
- 因结局选择视频片段或修改动态算法；
- 将 R2/M1 直接解释为真实生理波频率；
- 将 MSE 直接解释为已验证双壁协调性。
