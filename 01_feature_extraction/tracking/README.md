# Tracking

追踪源码已迁入。

正式运行使用：

```text
../frozen_runtime_20261126/analysis/mse_pregnancy_association/code/measurement/
```

其中包含 `_lk.py`、`geometry.py`、`tracking.py`、`deformation.py` 等。

历史审计快照保留在：

```text
../a0_wave_pipeline/uterine_wave/
```

A0 正式链保持原冻结逻辑：追踪、pose/刚体校正、连续有效段、约 2 s 径向测量；不跨无效段拼接。

本次迁移没有重新定义追踪阈值，也没有新增 outcome-driven QC。
