# Profiling 与 Partition 平台数据登记

## H100_SXM_80GB

- GPU：NVIDIA H100 80GB HBM3。
- 当前 kernel profiling、v3 性能建模以及 Partition 中现有性能模型均归属于该平台。
- 正式模型数据：`/output/Profiliing/artifacts/platforms/H100_SXM_80GB/model_ready_v3`。
- lowThrust_13 补采也属于 H100。

## RTX_PRO_6000_BLACKWELL_SERVER_EDITION

- GPU：NVIDIA RTX PRO 6000 Blackwell Server Edition。
- 当前只有 10 个完整矩阵的规则 NB 实验，以及 3 个矩阵的 24576/32768 边界探测。
- 主候选 NB：1024、2048、4096、8192、16384。
- 这些是方案级/端到端 NB 数据，不是 H100 kernel profiling 训练标签。
- 数据：`/openbayes/home/Partition/artifacts/matrix16_selected11_fixed_nb_v1/rtx_pro6000_labeled/runs_labeled_rtx_pro6000.csv`。

## 强制隔离规则

1. 每条 GPU 时间标签必须带 `hardware_id` 和 `gpu_model`。
2. 不同 GPU 的 runtime label 禁止合并训练；目录按 `artifacts/platforms/<hardware_id>/` 分开。
3. CPU feature replay 可以复用方法和代码，但每个平台的 GPU CSV 都必须独立严格连接，并校验 canonical hash/NNZ。
4. Partition 加载或导出的模型必须带相同 `hardware_id`；换平台必须重新 profiling、训练并导出。
5. RTX Pro 6000 的 NB 实验不得混入 H100 kernel 模型。
