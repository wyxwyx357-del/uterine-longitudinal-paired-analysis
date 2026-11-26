# R2 local dominant frequency

R2 是独立测量链：

```text
背景刚体校正后的局部组织位移
→ 局部解剖区域
→ 0.5–5 cpm 谐波频率拟合
→ dominant frequency
```

它不由 amplitude、ER、MSE 或 M1 计算得到。

## 当前状态

冻结源码 Python 文件本体目前没有在可访问 GitHub 仓库或文件库中找到，因此**尚未迁入源码**。

已经恢复到当时的源码 SHA256 清单，保存在：

`EXPECTED_SOURCE_HASHES.json`

其中包括：

- ab_block_tracking.py
- frequency.py
- rigid_correction.py
- run_local_motion_frequency.py
- run_regional_frequency.py
- validate_regional_v1_1.py
- 以及其它当时冻结文件

后续拿到 R2 代码后必须逐文件校验 SHA，再迁入；不根据审查报告重新实现以冒充冻结源码。
