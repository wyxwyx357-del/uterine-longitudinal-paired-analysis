# R2 frequency

R2 完整冻结主链已迁入：

```text
../frozen_runtime_20261126/r2_frozen/
```

版本：

```text
R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106
```

该目录包含 22 个冻结文件，包括：

- CUDA sparse LK
- block tracking
- rigid correction
- local longitudinal/tangential motion
- 0.5–5.0 cpm harmonic frequency fitting
- regional frequency
- longitudinal frequency
- reproducibility checks
- validated-stable-medium consensus
- batch runner

迁移后已逐文件做 Git blob 比对：**22/22 与冻结源字节一致。**

R2 与 A0 径向链保持独立。最终患者级 R2 仍按冻结规则从后壁保留区域聚合，不改成 M1 或临床蠕动次数。
