# Tracking

已迁入现有 P3 / radial tracking 源码。

## 来源

`wyxwyx357-del/uterine-tracking`

该仓库记录的原始迁移基线为：

`wyxwyx357-del/end@a9cf11b95293e395e0a90a1e3340393d782a2d5d`

## 代码

`peristalsis_pipeline/` 包含原样迁移的底层算法：

- tracking_lk.py
- tracking_mesh.py
- radial_pair_geometry.py
- tracking_huang_fusion.py
- tracking_huang_wall_fusion.py
- tracking_huang_radial_pairs.py
- p3_image_boundary_correction.py
- p3_anatomical_position_qc.py
- tracking_artifact_qc_v2.py

`entrypoints/` 保留原运行/复核入口。

来源 SHA 清单见 `MIGRATION_MANIFEST_SOURCE.md`。
