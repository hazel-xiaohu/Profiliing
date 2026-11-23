# PanguLU 最小 Launch 配置敏感性实验 v1

## 1. 目的和边界

本实验验证：在算法、输入稀疏结构和 GPU 平台不变时，合理的 CUDA launch-policy 扰动是否改变四类 kernel 的运行时间排序。当前只保留 `SSSSM-dense`；GETRF、GESSM、TSTRF 和 SSSSM-dense 在同一次 dense 运行中采集。

本目录与正式 v3 数据隔离。任何输出都不得拼接到 `artifacts/online_safe_v3/training_v3_strict`。

第一阶段记录配置编号、二进制 SHA256、严格任务键和实测 `time_kernel_ms`。要形成完整的论文证据，正式 profiling 还需按第 7 节在 C/CUDA catalog 中记录每次实际 launch 的 grid/block/resource 属性。

## 2. 准备三套可执行文件

配置见 `configs.example.json`：

- `baseline`：当前生产配置（2 warps/block）。
- `low`：合法的 warp-level 分支使用 1 warp/block。
- `high`：合法的 warp-level 分支使用 4 warps/block。

当前常量位于 `src/pangulu_cuda.h`。正式实现时先把四个常量改成可覆盖形式，默认值仍为 2：

```cpp
#ifndef WARP_PER_BLOCK_GEMM
#define WARP_PER_BLOCK_GEMM 2
#endif
#ifndef WARP_NUM_WARPLU
#define WARP_NUM_WARPLU 2
#endif
#ifndef WARP_PER_BLOCK_DGESSM
#define WARP_PER_BLOCK_DGESSM 2
#endif
#ifndef WARP_PER_BLOCK_DTSTRF
#define WARP_PER_BLOCK_DTSTRF 2
#endif
```

随后分别编译并保存为三个不同文件。每次必须清理 CUDA 对象，不能只重新链接。构建时记录：Git commit、完整 `NVCCFLAGS`、CUDA/driver 版本、GPU 型号。`build_config_variants.sh` 当前只检查/列出目标，不会在有大量未提交修改的工作树里自动改源码或清理构建产物。

注意：GESSM/TSTRF 的较密分支固定为 256 threads，GETRF 也有多个自适应分支。第一阶段不得为了凑齐配置而强行改变与实现语义绑定的线程数；应记录实际分支，并将未扰动分支标为 `not_perturbed`。

## 3. 选择最小矩阵集

复制模板并填写 4～6 个已下载矩阵的绝对路径：

```bash
cp experiments/launch_sensitivity_v1/matrices.example.txt \
   experiments/launch_sensitivity_v1/matrices.local.txt
```

矩阵应覆盖：低/中/高密度、较规则/不规则结构，以及 tiny/regular/large kernel 任务。无需重新下载，也无需跑完整 24 矩阵。

## 4. 运行

先复制并修正三个二进制路径：

```bash
cp experiments/launch_sensitivity_v1/configs.example.json \
   experiments/launch_sensitivity_v1/configs.local.json
```

检查任务计划，不启动求解：

```bash
python3 experiments/launch_sensitivity_v1/run_launch_sensitivity.py \
  --configs experiments/launch_sensitivity_v1/configs.local.json \
  --matrix-list experiments/launch_sensitivity_v1/matrices.local.txt \
  --output artifacts/launch_sensitivity_v1 \
  --block-sizes 2048 4096 6144 8192 10240 \
  --repeats 10 --warmups 1 --dry-run
```

正式运行：

```bash
python3 experiments/launch_sensitivity_v1/run_launch_sensitivity.py \
  --configs experiments/launch_sensitivity_v1/configs.local.json \
  --matrix-list experiments/launch_sensitivity_v1/matrices.local.txt \
  --output artifacts/launch_sensitivity_v1 \
  --block-sizes 2048 4096 6144 8192 10240 \
  --repeats 10 --warmups 1 --timeout 1800
```

断点续跑时加 `--resume`。脚本只运行 dense variant，并保留 `inf/nan` residual，不设有限残差阈值。

## 5. 输出与严格配对

每个配置产生：

```text
artifacts/launch_sensitivity_v1/configs/<config>/
├── campaign/                 原始运行和 catalog
└── measurements/
    ├── GETRF.csv
    ├── GESSM.csv
    ├── TSTRF.csv
    └── SSSSM.csv             仅 dense
```

脚本为每行增加：

```text
launch_config_id, repeat_index, profiling_run_id,
matrix_path, source_catalog, binary_sha256
```

分析：

```bash
python3 experiments/launch_sensitivity_v1/analyze_launch_sensitivity.py \
  artifacts/launch_sensitivity_v1
```

分析以 `kernel_type + task_id + 矩阵/层/块/配置字段` 连接，并逐项检查：

```text
structure_hash
nnz_input_0/1/2
nnz_output_0/1
```

不一致任务写入 `analysis/rejected_pairs.csv`，不参与统计。每任务先取 10 次测量的中位数，再计算配置间 speedup、绝对变化和 Spearman 排序相关系数。

## 6. 判定标准

最低可接受证据：

- 每类 kernel 的配对任务数充分，且不存在系统性结构拒绝；
- 配置间 runtime 排序 Spearman ≥ 0.90；
- 分尺寸/密度检查没有明显的系统性排序反转；
- 后续将各配置模型用于同一批 Partition 候选，报告 Top-1、Top-2、最终分块点一致率和 regret。

本脚本的 runtime Spearman 不能单独证明分块决策稳定。决策验证必须固定同一候选集合，分别替换模型输出后执行。

## 7. 后续正式 profiling 必须新增什么

### 必须新增

1. **配置身份和复现信息**

   `launch_policy_id`、二进制 SHA256、Git commit、完整编译参数、CUDA runtime/driver、GPU 名称和 compute capability。不同硬件仍分别 profiling 和训练，不混成一个跨架构模型。

2. **每次实际 launch 的配置**

   `launch_branch`、`grid_{x,y,z}`、`block_{x,y,z}`、`dynamic_shared_bytes`。必须记录实际走到的 GETRF/GESSM/TSTRF 自适应分支，而不是只记录期望配置。

3. **kernel 静态资源属性**

   通过 `cudaFuncGetAttributes()` 记录 `numRegs`、`sharedSizeBytes`、`localSizeBytes`、`maxThreadsPerBlock`。这些信息应按 kernel symbol/config 缓存查询，不能每个任务重复做昂贵查询。

4. **设备及派生并发属性**

   记录 `sm_count`；用 `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` 得到 `active_blocks_per_sm`，派生 `theoretical_occupancy` 和 `waves_per_sm`。它们是配置产生的观测量，不应伪装成互相独立的控制变量。

5. **数据质量字段**

   保留 `task_id + 配置字段`、`structure_hash`、全部输入/输出 NNZ、repeat、warmup、运行顺序、失败/超时状态、CUDA error。正式合并仍应结构不一致即拒绝。

6. **阶段总时长**

   campaign wall-clock、warmup、measured GPU runs、catalog I/O、聚合、feature replay、严格合并、训练、五折交叉验证分别计时，并输出机器可读 JSON/CSV 和 Markdown 报告。

### 建议新增

- 固定或记录 GPU application clock、功耗模式、温度区间；至少记录 `nvidia-smi -q` 快照。
- 每组配置随机化任务执行顺序，但三配置使用相同 seed；保留顺序编号以检查热漂移。
- 每个配置独立 warmup；推荐 1 次 warmup、10 次正式重复并用中位数作为标签，同时保存离散度。
- 在建模前报告重复测量 CV/MAD，剔除逻辑必须透明且不按模型误差删样本。
- 只训练并部署 SSSSM-dense；SSSSM-sparse 历史数据只作历史对照。

### 第一轮不需要新增

- 不需要把测得 runtime、cache miss、运行后计数器作为在线输入特征；
- 不需要边运行边自动训练；
- 不需要为了寄存器敏感性立即使用 `-maxrregcount`；
- 不需要独立任意改变 grid，grid 必须保持任务覆盖关系；
- 不需要重新做 feature replay：launch 配置不改变符号结构，已有结构特征可按严格任务键复用。

## 8. 对正式模型的建议

正式 profiling 的主模型仍然使用运行前可获得的输入结构特征。launch/hardware 字段有两种合理用法：

1. 若生产 launch policy 固定：为每个 GPU 平台和 policy 单独 profiling/训练，launch 字段主要用于审计与复现，不必作为在线特征。
2. 若在线阶段会在多个 launch policy 中选择：把候选 policy 的 grid/block/waves/occupancy 作为候选配置特征，模型预测每个候选时间再选择；此时必须保证这些字段在 kernel 启动前可计算。

当前研究目标更适合第 1 种：换硬件后重新 profiling，而不是训练一个覆盖所有 GPU 架构的统一模型。
