# PanguLU GPU Profiling 与性能建模操作指南

本文档说明将当前代码迁移到 GPU 平台后，如何完成环境适配、编译、正确性验证、CUDA Event 数据采集和性能建模。

## 0. 当前里程碑

状态快照：2026-07-28。旧数据已隔离到 `artifacts/legacy_postkernel_v1/`；下一轮在线安全采集统一写入 `artifacts/online_safe_v2/`。旧数据仅用于历史结果追溯，不得与新 schema 合并。

### Milestone 1：GPU kernel catalog 采集已完成

- GETRF、GESSM、TSTRF、SSSSM 已按任务类型分别输出四份 CSV。
- SSSSM sparse/dense 写入同一类 SSSSM catalog，并由
  `kernel_variant` 和 `gpu_kernel_name` 区分。
- `time_kernel_ms` 使用 CUDA Event 测量；同时保留端到端时间、预处理时间、
  kernel 启动次数及任务结构特征。
- H100 平台已完成编译和基本冒烟验证。

### Milestone 2：SuiteSparse 候选集与自动 campaign 已完成

- `matrix/profiling_candidates.csv` 当前包含 25 个矩阵。
- 候选矩阵来自 25 个不同 Group，避免同一应用场景或高度相似矩阵重复；
  行数均不超过 1,000,000。
- 下载器支持 SuiteSparse CSV 清单、缓存、重试、安全解包和 Matrix Market
  类型校验；当前接受 numeric real/integer，拒绝 complex 和 pattern/binary。
- `script/run_profiling_suite.py` 已支持预热、断点续跑、超时、强制
  sparse/dense、原始日志、成功/失败标记及按矩阵聚合四份 CSV。

当前运行目录为 `artifacts/legacy_postkernel_v1/campaign`，配置如下：

```text
25 matrices
× 5 block sizes: 1024, 2048, 4096, 8192, 16384
× 2 SSSSM variants: sparse, dense
= 250 profiling groups

repeats = 1
warmups = 1
max_residual = unset
GPU = NVIDIA H100 80GB HBM3
```

本轮 campaign 已结束，最终状态为：

```text
250 / 250 warmup groups attempted
200 / 250 groups passed warmup and completed measured profiling
20 / 25 matrices produced aggregated results
200 successful measured runs
0 measured-run _FAILURE.json markers
```

`run_status.csv` 只包含进入正式采集阶段的 200 个成功运行。其余 50 个分层
在 warmup 阶段被拦截，因此不会生成 measured-run `_FAILURE.json`。历史 campaign 包含 20 个有效矩阵。为避免三份数据重复，`aggregated/` 已删除；唯一原始证据保留在归档的 `campaign/raw/`，需要时可重新聚合。

### Milestone 3：训练集抽样与性能模型流水线已完成

- `script/build_kernel_training_sets.py` 已实现按“矩阵 × block size”分层、
  无放回随机抽样。
- 目标规模为 GETRF 10,000、GESSM 20,000、TSTRF 20,000、SSSSM
  40,000 行。
- SSSSM 的每一行对应完全相同的 320 个任务之一，同时包含 sparse 和 dense
  的实测 `time_kernel_ms`，不会把两种 variant 当成互不相关的样本。
- `script/train_kernel_time_models.py` 已实现 GETRF/GESSM/TSTRF 独立回归模型，
  以及基于相同特征的 SSSSM sparse/dense 双模型和 kernel selector。
- 交叉验证按 `matrix_id` 分组，避免同一矩阵的数据泄漏到训练集和验证集两侧；
  输出误差指标、逐样本预测、选择准确率和 regret。

### Milestone 4：阶段性模型、Roofline 和可视化已完成

该阶段性快照已在 2026-07-28 清理，因为最终 200-run 与 Phase-1 结果已经覆盖它。以下指标仅保留为历史记录。该快照基于
188 个当时已完成的 measured runs，从 15 个矩阵中选出 29 个同时满足全部
配额的“矩阵 × block size”分层，无放回生成：

```text
GETRF       2,320 rows
GESSM       4,640 rows
TSTRF       4,640 rows
SSSSM       9,280 paired sparse/dense rows
replacements    0 rows
```

SSSSM 非线性 selector 的 matrix-grouped CV 准确率为 91.5625%，mean regret
为 6.0122%，p90 regret 为 0。可解释 Roofline 中仅 SSSSM dense 当前标记为
`usable_preliminary`，grouped-CV R² 为 0.9202；GETRF、GESSM、TSTRF 和
SSSSM sparse 因带宽参数不可辨识或泛化不足标记为 `diagnostic_only`。

已生成以下可复现产物：

```text
training_set/       分层抽样 CSV 与 sampling manifest
models/             HGB 模型、CV 预测和 selector 指标
roofline_models/    闭式公式、参数、可信度警告和 CV 预测
roofline_figures/   实测-预测图、算术强度图（PNG/PDF）
summary.json        本轮阶段性实验摘要
```

### Milestone 5：基于最终 200 runs 的正式模型已生成

正式结果保存在 `artifacts/legacy_postkernel_v1/analysis/final_200runs/`。采用无放回策略，只保留
同时完成 sparse/dense 且满足 GETRF 80、GESSM/TSTRF 160、SSSSM 配对 320
要求的分层：

```text
source successful runs   200
selected matrices         15
selected matrix/block strata 32
GETRF samples          2,560
GESSM samples          5,120
TSTRF samples          5,120
SSSSM paired samples  10,240
replacement samples        0
```

HGB matrix-grouped 5-fold CV 结果：

```text
GETRF          R² 0.6206, MAPE 24.19%
GESSM          R² 0.4848, MAPE 39.78%
TSTRF          R² 0.7926, MAPE 36.93%
SSSSM sparse   R² 0.7191, MAPE 53.92%
SSSSM dense    R² 0.7436, MAPE 14.73%
SSSSM selector accuracy 91.5234%
SSSSM selector mean regret 5.9207%, p90 regret 0
```

正式 Roofline 中 GETRF（CV R² 0.6971）与 SSSSM dense（CV R² 0.9179）标记
为 `usable_preliminary`；GESSM、TSTRF、SSSSM sparse 仍为
`diagnostic_only`。训练 CSV 行数、重复标记、5 个 joblib 模型及两张 PNG
均已完成完整性检查。

历史复现已冻结：结果、训练集和模型均保存在归档中。当前在线安全训练器会主动拒绝 v1 CSV；如需逐指令复现旧指标，应使用归档时对应的旧代码版本，不要把 v1 数据传给当前训练器。

### Milestone 6：零 GPU 重跑的第一阶段模型改进已完成

第一阶段结果保存在 `artifacts/legacy_postkernel_v1/analysis/phase1_weighted/`。数据来自现有
200 个成功 runs，不启动 GPU；保留 20 个矩阵、100 个完整 sparse/dense
分层和全部 5 种 block size。每层最多抽取既定配额，样本不足时保留全部数据，
并用逆分层样本数加权；没有重复抽样。

```text
GETRF rows        4,533
GESSM rows       10,503
TSTRF rows       10,503
SSSSM paired     23,262
```

该历史模型使用 `log(time_ms + 1e-4)` 目标、variant-specific launch count、per-launch 特征和 5×3 nested matrix-grouped CV；这些运行时特征现已判定不适用于在线决策，因此该模型只保留作历史基线。与相同数据上的旧模型公平对比：

```text
Kernel          old MAPE    phase1 MAPE
GETRF             20.41%         19.30%
GESSM             48.16%         36.93%
TSTRF             44.73%         32.79%
SSSSM sparse      55.03%         38.07%
SSSSM dense       15.96%         14.44%
```

SSSSM 独立时间 selector 的准确率由 91.80% 提升到 94.37%，mean regret
由 5.47% 降到 3.14%。regret-weighted ratio selector 的准确率为 93.71%、
mean regret 为 3.31%，未超过独立时间 selector，因此暂不用于替换。

历史复现已冻结：归档保留完整训练集、模型、CV 预测和分析报告。当前 schema-v2 流水线不接受这套旧 CSV。

产物包括 `training_set/`、`models/`、`baseline_same_coverage/` 和
`analysis/`；公平对比指标与图位于 `analysis/README.md`。

### Milestone 7：EBM 与 Roofline-residual 公平对比已完成

在与 Phase-1 HGB 完全相同的 20 个矩阵、100 个分层、分层权重和 5 折
matrix-grouped CV 上，训练了直接 EBM 以及“延迟感知 Roofline + EBM log-residual”模型。
依赖隔离在 `.venv-ebm`，版本固定于 `requirements-ebm.txt`。

```text
Kernel          HGB MAPE   EBM MAPE   Roofline-residual MAPE
GETRF              19.30%     21.71%                 21.37%
GESSM              36.93%     40.50%                 39.71%
TSTRF              32.79%     36.27%                 37.17%
SSSSM sparse       38.07%     41.52%                 41.49%
SSSSM dense        14.44%     16.82%                 16.94%
```

HGB 在五类 kernel 的 MAPE 都最优，SSSSM selector 准确率/mean regret 为
94.37%/3.14%，优于 EBM 的 93.67%/3.40% 和 Roofline-residual 的
93.72%/3.52%。因此部署选择仍推荐 HGB。

Roofline-residual 在长任务绝对尺度上具有优势：GETRF、SSSSM sparse、SSSSM
dense 的 grouped-CV R² 分别为 0.9694、0.9581、0.8879，WAPE 分别为
15.20%、24.10%、23.79%，明显优于对应 HGB 的部分绝对误差指标。建议论文中
同时报告 HGB 的相对误差/selector，以及 Roofline-residual 的物理解释与长任务表现。

EBM 全局重要性显示：GESSM/TSTRF 主要受 block size 和行/列不规则度影响；
SSSSM sparse 的关键项包含 structural factor、kernel launches 和 operational
intensity；SSSSM dense 主要由 FLOPs 和 bytes 相关项驱动。

结果目录：

```text
artifacts/legacy_postkernel_v1/analysis/phase1_weighted/
├── ebm_roofline_residual/   10 个模型、CV 预测、公式与指标
└── ebm_comparison/          公平对比报告、指标图、EBM 重要性图
```

历史 EBM/Roofline-residual 产物已完整归档。当前脚本使用在线安全特征集合，不能直接对 v1 训练集重跑并声称复现原指标。


### Milestone 8：旧数据归档与在线安全 schema v2 已建立

- 旧 campaign、训练集和模型统一归档到 `artifacts/legacy_postkernel_v1/`。
- 保留不可替代的 `raw/`、warmup 证据、200-run 正式结果和 Phase-1/EBM 对比。
- 删除 574 MB 可再生 `aggregated/`、旧 partial 快照以及已确认重复的顶层训练集/模型副本，总计约释放 620 MB。
- `manifest.json` 记录保留项、删除项、schema 局限和正式训练集 SHA-256。
- 新采集固定写入 `artifacts/online_safe_v2/campaigns/<run-name>/`，新模型写入 `artifacts/online_safe_v2/models/<run-name>/`。
- schema v2 在目标 kernel 前冻结 A/L/U/X 结构；模型不再使用 kernel launch count 或任何 per-launch 派生特征。旧 v1 CSV 禁止与 v2 拼接训练。

### Milestone 9：统一 NB 寻优与混合分块消融框架已建立

- `Partition` 当前采用基本粒度 `P=4096`、`MAX_NUM=5`，`ptr[]` 输出接口保持不变。
- symbolic 构造过程中同步生成任务摘要缓存，候选评估不再重复扫描完整 L/U。
- 已实现 `no_merge`、`greedy`、`dp` 和 `uniform_nb` 四种策略；其中 DP 是当前可加预测代价下的全局最优，不等同于 GPU 实测 oracle。
- 9 个完整矩阵的单次配对实验中，相对 Main 原生 NB，固定 `P=4096`、no-merge、Greedy、DP 的 pipeline 几何平均收益分别为 11.69%、15.71%、18.15%、19.22%。主要收益来自统一块大小变化，Greedy 和 DP 的额外收益较小。
- Greedy 相对 no-merge 的几何平均增益为 2.89%；DP 相对 Greedy 为 1.31%。Greedy 对 DP 的预测 regret 仅 0～0.37%，说明当前主要改进空间在代价函数，而不是搜索算法。
- 在线决策开销已经较低：Greedy/DP 中位约 2.8 ms，pipeline 中位占比约 0.021%。
- 新增 `uniform_nb`：枚举统一 NB，使用与混合分块相同的任务统计与 Roofline 代价，预测收益不足 5% 时回退到 4096。
- 由于现有 GPU 训练最大尺寸为 16384，默认候选为 `4096,8192,12288,16384`；只有补充更大尺寸训练数据后才应提高 `PANGULU_PARTITION_UNIFORM_MAX_NB`。
- `script/run_partition_quality_benchmark.py` 已加入 `partition_uniform_nb`，并可用 `--fixed-sweep 4096 8192 12288 16384` 生成统一 NB 实测 oracle。
- `rajat06` 冒烟测试选择 `NB=12288`，预测收益 67.734%，partition decision 总开销 0.364 ms，symbolic 输出的单块边界合法；其后出现的 `cuda malloc error` 位于数值阶段，不能作为端到端成功样本。

下一验收点：完成剩余 benchmark 配置并至少重复 3 次，用实测 uniform oracle 评价 `uniform_nb` 的选择 regret，再判断混合尺寸 Greedy/DP 是否提供了超出最佳统一 NB 的稳定收益。

### Milestone 10：正式 v3 建模统一为共享 matrix-grouped 5-fold

- 正式入口 `script/run_v3_shared_split_experiments.py` 不再使用矩阵内部 8:2，而是强制按矩阵分组的 5 折交叉验证。
- 五个目标共享 `splits/shared_matrix_folds_v3.csv`；分折使用多目标样本负载平衡，并由 seed 确定性复现。
- 消融 HGB、HGB、EBM、Analytical Roofline、Roofline-residual 全部使用相同折；交叉验证后再用全数据训练部署模型。
- 输出完整 OOF 预测、逐折指标以及 CSV/JSON/Markdown 三种训练耗时报告。
- 计时覆盖数据读取与校验、特征构造、每折 fit/predict、全量重训、模型/预测/汇总写出和总体 wall-clock。
- 现有五个 model-ready 数据集特征齐全，包含 23～24 个矩阵；SSSSM dense/sparse 258,481 行严格对齐。
- 旧 8:2 结果只保留为历史对照；新的正式训练必须写入空目录 `artifacts/online_safe_v3/experiments_grouped_5fold_v3`。

正式五折训练已完成。最佳 MAPE 分别为 GETRF 14.53%（Roofline-residual HGB）、GESSM 8.96%（HGB）、TSTRF 8.32%（HGB）、SSSSM-dense 12.74%（Roofline-residual HGB）、SSSSM-sparse 39.08%（HGB）。总 wall-clock 为 1727.04 s；正式指标见 `model_comparison_v3.csv`，耗时见 `training_timing_report_v3.md`。

### 当前已知问题与下一验收点

5 个矩阵、共 50 个分层没有通过 warmup：

- `Andrews`：历史 v1 的 10 个分层得到非有限残差 `inf`。
- `lhr71`、`mac_econ_fwd500`、`power197k`：历史 v1 各 10 个分层得到 `-nan`。
- schema v2 会识别并记录 `inf/-inf/nan/-nan`，但不会用它们触发 warmup 失败或参与 `--max-residual` 比较；进程非零退出、超时和真正缺少残差仍然失败。
- `Ga41As41H72`：10 个分层都在 symbolic/flops 输出后提前终止，没有进入
  preprocess；当前日志不能证明具体是 OOM，需要单独复现并采集退出码和资源峰值。

这些失败都发生在 measured profiling 之前，所以 `run_status.csv` 中看不到失败行。
非有限残差现在可以按上述规则忽略。`--allow-missing-residual` 仍只控制完全没有残差输出的情况，不能掩盖非零退出、超时、catalog 缺失或进程提前终止。

下一步是使用在线安全 schema v2 重新采集。旧 200-run 数据和全部历史模型只用于对照；新模型必须独立生成 availability、训练集与 grouped-CV 结果，禁止与旧 CSV 拼接。固定样本目标仍需根据新一轮 `sampling_availability.csv` 决定无放回保留策略。

## 1. 当前建模目标

主程序在 `src/` 中采集以下目标：

```text
GETRF        -> pangulu_getrf_cuda_kernel
GESSM        -> pangulu_gessm_fp64_cuda_v7
TSTRF        -> pangulu_tstrf_fp64_cuda_v7
SSSSM sparse -> pangulu_ssssm_cuda_kernel
SSSSM dense  -> pangulu_ssssm_dense_cuda_kernel
```

GETRF 只采集稀疏版本。纯对角块只执行复制，不启动 GETRF kernel，因此不会写入 GETRF 训练样本。

SSSSM 标定时必须对相同配置分别强制运行 sparse 和 dense。旧的密度阈值只能作为兼容的 `auto` 行为，不能用于重新标定选择边界。

## 2. 数据字段

启用 catalog 后，每项数值任务写入一条 CSV 记录。

```text
time_kernel_ms      CUDA Event 测得的目标 GPU 调用时间
time_total_ms       整个任务接口的端到端时间
time_preprocess_ms  GESSM/TSTRF 首次预处理时间
```

`time_total_ms` 可能包含数据搬运、格式转换、同步和 CPU 包装开销。纯 kernel 建模使用 `time_kernel_ms`，端到端任务建模使用 `time_total_ms`。

CUDA Event 只在设置 `CALIB_MODE=catalog` 时创建，普通运行不会产生 profiling Event。

## 3. 检查平台环境

```bash
cd /output/Profiliing
nvidia-smi
nvcc --version
mpic++ --version
mpirun --version
```

确认 GPU compute capability。正式 profiling 使用固定的架构/工具链契约：

```text
H100                    -> sm_90  -> CUDA 12.1（PTX 8.1）
RTX Pro 6000 Blackwell  -> sm_120 -> CUDA 12.8（PTX 8.7）
```

## 4. 适配构建配置

根目录 `make.inc` 已将 GPU 架构与 CUDA 工具链强绑定，禁止交叉组合：

```bash
make GPU_ARCH=sm_90    # 强制 /usr/local/cuda，CUDA 12.1
make GPU_ARCH=sm_120   # 强制 /output/toolchains/cuda-12.8，CUDA 12.8
```

对应的生成目标分别为：

```text
sm_90:  compute_90 + sm_90
sm_120: compute_120 + sm_120
```

`run_profiling_suite.py` 会在下载矩阵和启动 GPU 前读取可执行文件中的
PTX target/version。`H100_SXM_80GB` 只接受 `sm_90 + PTX 8.1`，
`RTX_PRO_6000_BLACKWELL` 只接受 `sm_120 + PTX 8.7`；不一致时直接终止，
避免错误标签再次进入正式数据集。

修正 METIS 静态库路径：

```make
METISFLAGS = <实际路径>/libmetis.a
```

当前工作区曾显示 `metis/` 下大量文件被删除。迁移后需要恢复内置 METIS 源码并构建，或者直接使用平台已有的 `libmetis.a`。

## 5. 编译

头文件修改较多，建议完整重编译：

```bash
make clean
make -B
ls -l test/PanguLU
```

## 6. 预热运行

选择一个较小矩阵，不开启 catalog，先运行一次：

```bash
mpirun -np 1 ./test/PanguLU \
  -F ./matrix/<matrix>.mtx \
  -O 32
```

预热用于初始化 CUDA context、加载模块，并确认矩阵读取和主流程正常。

## 7. 验证 SSSSM sparse/dense 正确性

强制 sparse：

```bash
PANGULU_SSSSM_KERNEL=sparse \
mpirun -np 1 ./test/PanguLU \
  -F ./matrix/<matrix>.mtx \
  -O 32
```

强制 dense：

```bash
PANGULU_SSSSM_KERNEL=dense \
mpirun -np 1 ./test/PanguLU \
  -F ./matrix/<matrix>.mtx \
  -O 32
```

两次运行必须正常结束并得到一致的求解结果。正式采集前，建议对小矩阵临时开启 `CHECK_ANS` 验证数值误差。

## 8. Profiling 冒烟测试

Sparse SSSSM：

```bash
rm -rf calib_smoke_sparse
mkdir -p calib_smoke_sparse

CALIB_MODE=catalog \
CALIB_CATALOG_DIR=calib_smoke_sparse \
PANGULU_SSSSM_KERNEL=sparse \
mpirun -np 1 ./test/PanguLU \
  -F ./matrix/<matrix>.mtx \
  -O 32
```

Dense SSSSM：

```bash
rm -rf calib_smoke_dense
mkdir -p calib_smoke_dense

CALIB_MODE=catalog \
CALIB_CATALOG_DIR=calib_smoke_dense \
PANGULU_SSSSM_KERNEL=dense \
mpirun -np 1 ./test/PanguLU \
  -F ./matrix/<matrix>.mtx \
  -O 32
```

必须使用不同的 `CALIB_CATALOG_DIR`。当前 catalog 以写模式打开，同一矩阵在同一目录中重复执行会覆盖旧文件。

每次运行会按任务类型生成四份文件：

```text
<matrix>.rank<rank>.GETRF.catalog.csv
<matrix>.rank<rank>.GESSM.catalog.csv
<matrix>.rank<rank>.TSTRF.catalog.csv
<matrix>.rank<rank>.SSSSM.catalog.csv
```

SSSSM 的 sparse 和 dense 数据都写入最后一份文件，并通过 `kernel_variant=sparse` 或 `kernel_variant=dense` 区分。

## 9. 检查 catalog

```bash
head -n 3 calib_smoke_sparse/*.catalog.csv
head -n 3 calib_smoke_dense/*.catalog.csv
```

核对标签：

```text
GETRF:        kernel_variant=sparse, gpu_kernel_name=pangulu_getrf_cuda_kernel
GESSM:        gpu_kernel_name=pangulu_gessm_fp64_cuda_v7
TSTRF:        gpu_kernel_name=pangulu_tstrf_fp64_cuda_v7
SSSSM sparse: kernel_variant=sparse, gpu_kernel_name=pangulu_ssssm_cuda_kernel
SSSSM dense:  kernel_variant=dense,  gpu_kernel_name=pangulu_ssssm_dense_cuda_kernel
```

核对时间：

```text
time_kernel_ms > 0
num_kernel_launches > 0
time_total_ms >= time_kernel_ms
```

如果 `time_kernel_ms=-1`，检查是否设置了 `CALIB_MODE=catalog`，以及任务是否进入目标 GPU 路径。

## 10. 批量采集

推荐使用可恢复的自动采集脚本：

```bash
python3 script/run_profiling_suite.py \
  --matrix-list profiling_matrices.txt \
  --block-sizes 32 64 128 256 \
  --repeats 10 \
  --warmups 1 \
  --output artifacts/online_safe_v2/campaigns/<run-name>
```

`profiling_matrices.txt` 每行一个 Matrix Market 文件路径，空行和以 `#`
开头的注释会被忽略。正式运行前可以增加 `--dry-run` 检查任务计划。
采集中断后使用相同参数并增加 `--resume`，已经通过校验的运行不会重复执行。

也可以直接读取与 `matrix/matrix16.csv` 相同格式的 SuiteSparse 清单并自动下载：

```bash
python3 script/run_profiling_suite.py \
  --suitesparse-csv matrix/matrix16.csv \
  --matrix-cache matrix/MM \
  --block-sizes 32 64 128 256 \
  --repeats 10 \
  --warmups 1 \
  --output artifacts/online_safe_v2/campaigns/<run-name>
```

清单必须包含 `Group`、`Name` 列。矩阵缓存路径为
`matrix/MM/<Group>/<Name>/<Name>.mtx`；再次运行会校验并复用缓存。
只下载和校验、不执行 profiling 时增加 `--download-only`，此时不需要
`--output`。

脚本会对每个矩阵和 block size 分别强制运行 sparse、dense SSSSM，保存
每次运行的原始 CSV、标准输出、残差和状态。最终输出：

```text
artifacts/online_safe_v2/campaigns/<run-name>/
├── campaign.json
├── run_status.csv
├── raw/
├── warmups/
└── aggregated/
    ├── <matrix-name>-<path-hash>/
    │   ├── matrix.json
    │   ├── GETRF.csv
    │   ├── GESSM.csv
    │   ├── TSTRF.csv
    │   └── SSSSM.csv
    └── all_matrices/
        ├── GETRF.csv
        ├── GESSM.csv
        ├── TSTRF.csv
        └── SSSSM.csv
```

每个矩阵目录都有独立的四份 CSV；目录名附加路径哈希，避免不同 Group
存在同名矩阵时互相覆盖。`all_matrices/` 提供跨矩阵总汇。GETRF、GESSM、
TSTRF 只从 sparse 实验中汇总，避免相同任务因 sparse/dense 两轮实验被
重复计数。SSSSM 同时汇总两轮实验，并由
`kernel_variant` 区分 sparse 和 dense。汇总数据还会附加
`profiling_run_id`、`repeat_index`、`matrix_path` 等溯源字段。

以下手动命令仍可用于调试单次运行。

每次重复使用独立目录，避免覆盖。

Sparse：

```bash
for repeat in $(seq 1 20); do
  output_dir="calib_sparse_r${repeat}"
  mkdir -p "$output_dir"

  CALIB_MODE=catalog \
  CALIB_CATALOG_DIR="$output_dir" \
  PANGULU_SSSSM_KERNEL=sparse \
  mpirun -np 1 ./test/PanguLU \
    -F ./matrix/<matrix>.mtx \
    -O 32
done
```

Dense：

```bash
for repeat in $(seq 1 20); do
  output_dir="calib_dense_r${repeat}"
  mkdir -p "$output_dir"

  CALIB_MODE=catalog \
  CALIB_CATALOG_DIR="$output_dir" \
  PANGULU_SSSSM_KERNEL=dense \
  mpirun -np 1 ./test/PanguLU \
    -F ./matrix/<matrix>.mtx \
    -O 32
done
```

建议覆盖多个矩阵、多种稀疏结构和多个 block size，例如 `32、64、128、256`。每组建议进行 10 至 20 次完整运行。

## 11. 建模

GETRF/GESSM/TSTRF 在两类 SSSSM 实验中执行路径相同，使用其中一组数据即可：

```bash
python3 script/perf_model_baseline.py \
  --target-field time_kernel_ms \
  --repeats 20 \
  'calib_sparse_r*/*.catalog.csv'
```

同时训练 SSSSM sparse/dense：

```bash
python3 script/perf_model_baseline.py \
  --target-field time_kernel_ms \
  --repeats 20 \
  'calib_sparse_r*/*.catalog.csv' \
  'calib_dense_r*/*.catalog.csv'
```

预期输出分组：

```text
GETRF:sparse
GESSM:default
TSTRF:default
SSSSM:sparse
SSSSM:dense
```

端到端任务建模：

```bash
python3 script/perf_model_baseline.py \
  --target-field time_total_ms \
  --repeats 20 \
  'calib_sparse_r*/*.catalog.csv' \
  'calib_dense_r*/*.catalog.csv'
```

## 12. 推荐执行顺序

```text
适配 CUDA/MPI/METIS
-> 完整编译
-> 小矩阵预热
-> sparse/dense 正确性验证
-> profiling 冒烟测试
-> 检查 Event 和 catalog
-> 多矩阵、多 block size、多次重复采集
-> 按 kernel_variant 分组建模
```

不要在标定 SSSSM 时使用旧的自动密度阈值。只有完成 sparse/dense 成对采集并得到新模型后，才应重新定义部署阶段的选择策略。

## 13. 分层抽样与 kernel time 建模

campaign 尚未结束时，可以用已经完成且满足全部配额的分层训练阶段性模型：

```bash
python3 script/run_partial_modeling.py artifacts/online_safe_v2/campaigns/<run-name>
```

脚本默认从同时完成 sparse/dense、且分别满足 GETRF 80、GESSM 160、TSTRF
160、SSSSM 配对 320 条要求的分层中无放回抽样，然后执行 5 折
matrix-grouped CV。结果写入带 UTC 时间戳的
`partial_modeling/<timestamp>/{training_set,models,roofline_models}/`，并生成
`summary.json`。`roofline_models/roofline_report.md` 给出可直接引用的闭式公式，
模型形式为：

```text
T_ms = L0 + max(F_GFLOP / P_eff_TFLOPS,
                B_GB / BW_eff_TBPS)
```

报告同时给出按矩阵分组的交叉验证指标、计算/访存瓶颈比例和参数不可辨识警告。
`roofline_figures/` 自动生成 PNG/PDF 两种格式的实测-预测图和经典
Roofline 算术强度图。拟合出的 `P_eff`、`BW_eff` 是任务路径有效吞吐，
不是 GPU 硬件峰值。
脚本不会读取尚未生成 `_SUCCESS.json` 的运行，也不会干扰仍在执行的 campaign。
自定义输出和标签示例：

```bash
python3 script/run_partial_modeling.py artifacts/online_safe_v2/campaigns/<run-name> \
  --output-root artifacts/online_safe_v2/models \
  --tag experiment_01
```

这是探索性模型；campaign 完成后仍需按下面流程重新生成正式训练集。

完整 profiling 结束后，按“矩阵 × block size”分层抽样：

```bash
python3 script/build_kernel_training_sets.py artifacts/online_safe_v2/campaigns/<run-name> \
  --output artifacts/online_safe_v2/models/<run-name>/training_set \
  --getrf 80 \
  --gessm 160 \
  --tstrf 160 \
  --ssssm 320
```

对于 25 个矩阵和 5 个 block size，目标输出为：GETRF 10,000 行、
GESSM 20,000 行、TSTRF 20,000 行、SSSSM 40,000 行。SSSSM 每行是
同一 `task_id + repeat_index + rank` 的配对测量，同时含
`sparse_time_kernel_ms` 与 `dense_time_kernel_ms`，不是两组互不对应的样本。

脚本先生成 `sampling_availability.csv`。任一分层样本不足时默认停止；若必须
保持上述固定行数，可在新的输出目录运行时增加 `--with-replacement`，所有
补抽行都会由 `sampled_with_replacement=1` 标记。

训练模型：

```bash
python3 script/train_kernel_time_models_phase1.py \
  artifacts/online_safe_v2/models/<run-name>/training_set \
  --output artifacts/online_safe_v2/models/<run-name>/hgb
```

GETRF、GESSM、TSTRF 各训练一个 `time_kernel_ms` 模型。SSSSM 使用完全
相同的任务特征分别训练 sparse/dense 两个耗时模型，运行时选择预测耗时较小
的一种。交叉验证按 `matrix_id` 分组，避免同一矩阵泄漏到训练和验证两侧。
输出包括 `.joblib` 模型、`metrics.json`、逐样本 CV 预测以及 SSSSM 选择准确率
和 regret。

## CPU-only feature replay（profiling 完成后执行）

`feature_replay` 用与正式求解相同的符号分解、预处理和任务 DAG 枚举运行前特征，但编译期跳过 GETRF、GESSM、TSTRF、SSSSM 数值 kernel、CUDA 初始化、GPU 预处理、结果校验及 overlap 线程。它强制单 MPI 进程，因此不会产生跨 rank 的数值通信依赖。

为避免覆盖正在使用的 `test/PanguLU`，独立构建到 `build/feature_replay/`：

```bash
bash script/build_feature_replay.sh
```

对候选矩阵及五种块大小回放：

```bash
python3 script/run_feature_replay.py \
  --suitesparse-csv matrix/profiling_candidates.csv \
  --matrix-cache matrix/MM \
  --block-sizes 1024 2048 4096 8192 16384 \
  --output artifacts/online_safe_v2/feature_replay \
  --resume
```

输出布局为 `<output>/<matrix-id>/bs_<NB>/catalog/`，其中包含四类 kernel CSV。静态行的 `profile_scope` 固定为 `feature_replay_pre_kernel`，`time_ms` 和其他运行时字段为 `-1`；这些文件只提供运行前特征，不能直接作为时间标签。脚本会检查四个 CSV 是否都生成、scope 是否正确、时间是否保持未知。
