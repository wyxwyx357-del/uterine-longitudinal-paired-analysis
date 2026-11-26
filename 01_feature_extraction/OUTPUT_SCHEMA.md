# Feature extraction output schema

`auto_features.csv` 必须做到：

- 一行 = 一次检查；
- `exam_id` 唯一；
- 与临床表 `检查ID` 一一对应；
- 特征失败保持空值；
- 不包含姓名、门诊号、身份证号、电话、视频实名路径等身份信息。

## 当前纵向分析会读取的字段

| field | meaning | priority |
|---|---|---|
| exam_id | 检查唯一ID | required |
| anterior_amplitude | 前壁幅度 | primary dynamic |
| posterior_amplitude | 后壁幅度 | primary dynamic |
| r2_frequency_cpm | R2局部主导频率候选 | exploratory |
| anterior_er | 前壁ER | exploratory |
| posterior_er | 后壁ER | exploratory |
| mse | 双壁MSE候选 | exploratory |
| m1_prop_rate_min_per_min | M1显著传播活动率候选 | exploratory |
| tracking_valid_fraction | 追踪有效比例 | QC |
| video_qc | 视频/追踪QC标签 | QC |
| pose_source | pose/校正来源 | provenance |

## 推荐同时保留的溯源字段

这些字段当前第二模块不会强依赖，但建议特征导出时保留：

- feature_pipeline_version
- tracking_version
- r2_version
- source_video_duration_s
- feature_extraction_status
- failure_reason
- source_hash / config_hash（如已有）

## 明确边界

临床人工记录的正向/反向蠕动频率不从本文件生成；它们属于临床表。

R2 与 M1 也不应互相替代：
- R2 是局部组织运动主导频率候选；
- M1 是显著传播活动候选率；
- 二者都不直接等同于已验证的真实生理蠕动波频率。
