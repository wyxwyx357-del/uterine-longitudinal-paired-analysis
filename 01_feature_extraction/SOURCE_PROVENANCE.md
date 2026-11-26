# Feature extraction source provenance

本目录中的算法源码以“迁移已有实现”为原则，不重新设计核心算法。

## 1. P3 / radial tracking

来源仓库：

- `wyxwyx357-del/uterine-tracking`
- 该仓库自身记录来源基线：`wyxwyx357-del/end@a9cf11b95293e395e0a90a1e3340393d782a2d5d`

迁入位置：

- `tracking/peristalsis_pipeline/`
- `tracking/entrypoints/`

核心迁入文件保持来源文本内容，包括：

- `tracking_lk.py`
- `tracking_mesh.py`
- `radial_pair_geometry.py`
- `tracking_huang_fusion.py`
- `tracking_huang_wall_fusion.py`
- `tracking_huang_radial_pairs.py`
- P3 boundary correction / anatomical QC / artifact QC
- tracking-only pipeline entrypoint

来源迁移清单保存在：

- `tracking/MIGRATION_MANIFEST_SOURCE.md`

## 2. A0 -> radial deformation / amplitude / ER / mode / candidate events

来源仓库：

- `wyxwyx357-del/uterine-wave-propagation-features`

迁入位置：

- `wave_features/uterine_wave/`

迁入实际 Python 包，包括：

- `geometry.py`
- `_lk.py`
- `tracking.py`
- `quality.py`
- `deformation.py`
- `amplitude.py`
- `spectrum.py`
- `modes.py`
- `patient.py`
- `radial_events.py`
- `candidate_evidence.py`
- `events.py`
- `wavefit.py`
- `io.py`
- `frozen_pipeline.py`
- `input_inventory.py`
- `controls.py`
- `cli.py`

冻结流水线入口：

- `wave_features/run_frozen_features.py`

该链与 P3 独立仓库有历史/接口关系，但不能把二者混成同一算法版本。

## 3. Frozen M1 / record MSE

最终源码来源分支：

- repository: `wyxwyx357-del/uterine-wave-propagation-features`
- branch: `m1-global-pairing-unseen-v1-source-20261126`

功能分类副本：

- `m1/frozen_source/`
- `er_mse/frozen_source/`

同时保留原包路径结构：

- `frozen_m1_mse_bundle/analysis/mse_pregnancy_association/code/`

这样不修改原源码中的 `analysis.mse_pregnancy_association...` import 路径。

其中包括：

- activity counting
- global event ownership
- normalized activity
- patient activity aggregation
- activity strength
- record MSE numeric core
- strict record-MSE measurement boundary

## 4. R2 local dominant frequency

R2 必须保持独立链，不能用本仓库的 ER/spectrum 或 M1 替代。

当前状态：

- 已确认历史源码文件名和 SHA256 清单；
- 当前 GitHub 仓库与可访问文件库中未找到对应 Python 源文件字节；
- 因此 **R2 源码尚未迁入**；
- `r2_frequency/EXPECTED_SOURCE_HASHES.json` 保存已知冻结哈希，供后续源码到位后逐文件校验。

不得根据报告/摘要重写 R2 代码来冒充原冻结实现。
