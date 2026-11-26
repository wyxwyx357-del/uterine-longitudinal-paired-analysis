# Feature extraction output schema

正式冻结流水线先生成：

```text
FINAL_DYNAMIC_FEATURES.csv
```

随后 `export_auto_features.py` 只做字段投影/重命名，生成：

```text
01_feature_extraction/outputs/auto_features.csv
```

导出阶段不重新计算特征、不补缺失、不读取妊娠结局。

## 纵向分析主字段

| field | source in frozen table | meaning |
|---|---|---|
| exam_id | case_id | 检查唯一 ID |
| anterior_amplitude | anterior_amplitude | 前壁幅度 |
| posterior_amplitude | posterior_amplitude | 后壁幅度 |
| r2_frequency_cpm | R2_frequency_cpm | R2 局部主导频率候选 |
| anterior_er | anterior_ER_mean_paired | 前壁 ER |
| posterior_er | posterior_ER_mean_paired | 后壁 ER |
| mse | record_MSE_time_equal | 双壁 MSE 候选 |
| m1_prop_rate_min_per_min | m1_prop_rate_min_per_min | M1 显著传播活动候选率 |
| tracking_valid_fraction | raw_valid_fraction | A0 原始有效支持比例 |
| video_qc | 仅上游明确提供时使用 | QC |
| pose_source | 仅上游明确提供时使用 | provenance |

`video_qc` 和 `pose_source` 不从其他状态字段臆造；没有真实来源时保持缺失。

## 同时保留的状态/溯源

导出脚本会尽可能保留：

- pipeline_version
- a0_status
- anterior_amplitude_status
- posterior_amplitude_status
- mse_status
- mse_decomposition_status
- m1_status
- m1_failure_reason
- m1_algorithm_version
- R2_status
- R2_failure_reason
- R2_retained_posterior_region_count
- MSE decomposition fields
- M1 effective observation support

## 数据规则

- 一行 = 一次检查；
- `exam_id` 必须唯一；
- 特征失败保持 NaN；
- 不用 0 代替失败；
- 不包含患者直接身份信息；
- 临床人工正向/反向蠕动频率来自临床表，不由本模块生成。

## 解释边界

- R2：局部组织运动主导频率候选；
- M1：显著传播活动候选率；
- ER/MSE：影像信号层面的方向/协调性表征；
- 上述指标均不自动等同于已验证真实生理蠕动波。
