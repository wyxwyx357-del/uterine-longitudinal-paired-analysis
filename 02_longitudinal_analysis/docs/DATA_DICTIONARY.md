# De-identified analysis data dictionary

真实数据在本地整理成去标识化表后再运行代码。不要把真实数据提交到 Git。

## 必需字段

| canonical column | meaning |
|---|---|
| patient_id | 随机研究患者 ID，不得使用姓名/门诊号 |
| exam_id | 随机检查 ID；同一患者每次检查唯一 |
| exam_order | 患者内检查顺序：1, 2, 3, ... |
| video_duration_s | 分析视频可用时长（秒） |

## 推荐时间字段

| column | meaning |
|---|---|
| exam_date | 检查日期；如存在重识别风险，不进入公开数据 |
| days_from_first_exam | 相对第一次检查的天数，可用于替代真实日期 |
| days_to_transfer | 检查距移植的相对天数 |

`exam_order` 是主分析所需字段。患者有多少次检查，就按真实时间顺序编号多少次，不限制为 1/2。

## 结局

| column | meaning |
|---|---|
| clinical_pregnancy | 1=临床妊娠，0=未临床妊娠，缺失保持 NaN |

## 内膜

| column | meaning |
|---|---|
| endometrium_thickness_mm | 内膜厚度 |
| endometrium_volume_ml | 内膜容积 |
| endometrium_type | 内膜分型 |

## 血流

| column | meaning |
|---|---|
| vi | VI |
| fi | FI |
| vfi | VFI |
| pi | PI |
| ri | RI |
| sd | S/D |

## 临床蠕动

| column | meaning |
|---|---|
| clinical_c2f_rate_min | 临床记录的一种方向蠕动频率，单位 次/分 |
| clinical_f2c_rate_min | 临床记录的另一方向蠕动频率，单位 次/分 |
| clinical_total_peristalsis_rate_min | 两方向频率之和；仅当两项均有定义时计算 |

临床方向标签与自动算法 C2F/F2C 的解剖对应关系未确认时，不应强制映射。

## 自动动态特征

| column | meaning |
|---|---|
| anterior_amplitude | 前壁幅度 |
| posterior_amplitude | 后壁幅度 |
| r2_frequency_cpm | R2 局部主导频率候选 |
| m1_prop_rate_min_per_min | M1 显著传播活动率候选 |
| anterior_er | 前壁 ER |
| posterior_er | 后壁 ER |
| mse | 双壁 MSE 候选 |

## 可选混杂/质量字段

建议本地保留：
- treatment_protocol
- embryo_stage
- embryo_quality
- embryo_count
- tracking_valid_fraction
- video_qc
- pose_source

## 数据导出原则

用于代码分析的表中不得包含：
- 姓名；
- 门诊号；
- 身份证号；
- 电话；
- 住址；
- 原始视频路径中可识别姓名的部分；
- 其他直接身份信息。
