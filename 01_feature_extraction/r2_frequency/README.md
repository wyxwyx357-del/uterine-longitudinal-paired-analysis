# R2 local wall-tangent dominant frequency

R2 是独立测量链，不由 amplitude、ER、MSE 或 M1 推导。

```text
已有 video + R2 tracking seed + nodes + pixel scale
→ R2 re-tracking
→ background rigid correction
→ cervix-to-fundus local wall-tangent displacement
→ 0.5–5 cpm candidate fitting
→ technical/reproducibility rules
→ retained posterior regions
→ R2_frequency_cpm
```

## Current formal implementation

真正的 current R2 源码位于：

```text
../frozen_runtime_20261126/r2_frozen/
```

不要在本目录再维护第二套复制源码。

版本：

```text
R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106
```

对应锁文件：

```text
../frozen_runtime_20261126/r2_frozen/FROZEN_SPEC.json
../frozen_runtime_20261126/SOURCE_LOCK.json
```

## Current primary definition

- motion direction: local wall tangent, cervix → fundus
- frequency band: 0.5–5.0 cpm
- grid step: 0.05 cpm
- formal wall: posterior only
- primary regions: high confidence + independently validated stable medium
- spatial regions: cervical / middle / fundal
- exam-level aggregation: equal mean over retained posterior regions

正式检查级字段：

```text
R2_frequency_cpm
```

如果没有任何 retained posterior region：

```text
R2_frequency_cpm = NaN
R2_status = no_retained_posterior_region
```

失败不写成 0 cpm。

## Not the historical three-window statistic

当前纵向项目的正式 `R2_frequency_cpm` **不是**：

- anterior+posterior 六区中位数；
- 三个60秒窗口频率中位数；
- 最长连续段统计量；
- M1活动率；
- 临床人工蠕动次数。

历史定义可用于旧研究结果解释，但不能与 current R2 字段混用。

## Historical requested SHA audit

此前恢复过一组 historical requested SHA，但没有完整找回。那组 SHA 不是 current production lock。

历史审计记录保留为：

```text
HISTORICAL_REQUESTED_SOURCE_HASHES.json
```

current workflow 的权威来源锁始终是：

```text
../frozen_runtime_20261126/SOURCE_LOCK.json
```

## Outcome blindness

R2/current dynamic measurement chain 不读取妊娠结局，不以 AUC 或结局优化频率阈值。
