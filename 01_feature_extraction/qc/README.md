# QC

正式 QC 分散保留在各冻结链内部，不重新合并成新的算法阈值。

统一导出中保留：

- `tracking_valid_fraction`
- A0 / amplitude / MSE / M1 / R2 status
- failure reason
- R2 retained region count
- MSE decomposition status
- M1 effective observation support

失败保持缺失，不转换成 0。

`video_qc` 和 `pose_source` 只有在真实上游明确提供时才填写；导出脚本不会从其他状态字段臆造。
