# ER / MSE

## ER / mode window

实际源码：

```text
../wave_features/uterine_wave/spectrum.py
../wave_features/uterine_wave/modes.py
../wave_features/uterine_wave/patient.py
```

## Frozen record MSE

最终冻结数值核心已迁入：

```text
frozen_source/record_mse_core.py
frozen_source/record_mse_measurement.py
```

为保持原 import 路径，另保留：

```text
../frozen_m1_mse_bundle/analysis/mse_pregnancy_association/code/
```

MSE 是 ER 双壁协调性候选，不直接称为已验证生物标志物。
