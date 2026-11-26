# Source provenance

本文件记录本仓库特征提取代码的来源，防止迁移后混淆版本。

## 1. 正式运行版本

正式运行目录：

```text
frozen_runtime_20261126/
```

来源仓库：

```text
wyxwyx357-del/uterine-wave-propagation-features
```

来源分支：

```text
final-dynamic-batch-20261126
```

完整流水线版本：

```text
UTERINE_DYNAMIC_FINAL_BATCH_20261126_V1
```

迁入的 `SOURCE_LOCK.json` 与原分支 Git blob SHA 完全一致：

```text
a3c5ec2fea94ac4c982f89e16b47fb6246f07223
```

正式入口 `run_final_batch.py` 原 Git blob：

```text
c0990fad6dc498ea1c57032f44fe9d3df1ae8ad5
```

迁移后已核对一致。\n\n按 `SOURCE_LOCK.json` 反向检查正式 runtime 的 Git 锁定源码：\n\n```text\nexpected locked files: 34\nmissing:               0\nmismatch:              0\nbyte-exact:             yes\n```\n
## 2. M1

冻结来源提交：

```text
75d26aca0803c15535383b8a62b372c10a521863
```

算法版本：

```text
m1-global-pairing-unseen-v1
```

除了正式 runtime 中的依赖，还在：

```text
m1/frozen_source/
```

保留独立来源快照。

## 3. R2

版本：

```text
R2_LONGITUDINAL_POSTERIOR_FROZEN_CANDIDATE_V1_1_20261106
```

完整冻结目录：

```text
frozen_runtime_20261126/r2_frozen/
```

包含 22 个正式冻结文件。

迁移方法不是人工重写：原文件按原始字节创建 Git blob，再与本地冻结源的 Git blob SHA 逐项比对。

核验结果：

```text
expected files: 22
found files:    22
missing:        0
mismatch:       0
byte-exact:     yes
```

本地冻结源码包 SHA256：

```text
47e478d4ca3d2ff8c0ee5b3e1ff37a4635e1484a9c014663875e0840bf014906
```

该 SHA 仅用于来源审计；仓库公开内容不包含患者数据。

## 4. 历史 A0 审计快照

目录：

```text
a0_wave_pipeline/
```

来源提交：

```text
3b6e0541766849a13ff6e0de37cdf011695e04e8
```

该目录保留为历史审计/对照快照，不应替代 `frozen_runtime_20261126/` 作为当前正式一键运行入口。

## 5. ER/MSE

正式运行实现位于 `frozen_runtime_20261126/`。

另外在：

```text
er_mse/frozen_source/
```

保留 patient-level MSE 核心来源文件，便于单独审计。

## 6. 迁移边界

本次迁移：

- 不修改算法公式；
- 不改变冻结阈值；
- 不读取妊娠结局；
- 不上传患者数据；
- 不新增 pose 算法；
- 不把失败值改成 0；
- 只新增仓库组织、运行包装和字段投影脚本。


## 7. 静态迁移检查

在不读取患者数据的条件下完成：

- 冻结 runtime Python 源码 `compileall`：PASS；
- 按新仓库嵌套目录模拟 `PYTHONPATH` 后，`run_final_batch.py --help`：PASS；
- R2 `track_and_cache_frequency.py --help`：PASS；
- R2 `extract_longitudinal_frequency.py --help`：PASS。

这只证明迁移后的源码结构/导入可用，不等于已经在真实患者视频上完成端到端数值复现。正式批处理仍应在原 CUDA OpenCV 环境和允许使用的数据上执行。
