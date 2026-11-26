# Radial motion

径向运动/形变源码已迁入正式 runtime：

```text
../frozen_runtime_20261126/analysis/mse_pregnancy_association/code/measurement/deformation.py
../frozen_runtime_20261126/analysis/mse_pregnancy_association/code/measurement/amplitude.py
```

主要从冻结 A0 轨迹建立 (A(s,t)) / interval strain 表征，并输出前壁、后壁 amplitude 等检查级指标。

正式主测量间隔保持约 2 s。失败或不连续支持不做插值补齐。
