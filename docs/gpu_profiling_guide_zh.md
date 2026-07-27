# PanguLU GPU Profiling 与性能建模操作指南

本文档说明将当前代码迁移到 GPU 平台后，如何完成环境适配、编译、正确性验证、CUDA Event 数据采集和性能建模。

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

确认 GPU compute capability。常见配置：

```text
A100 -> sm_80
H100 -> sm_90
```

## 4. 适配构建配置

检查根目录下的 `make.inc`。

CUDA 路径应与平台一致，例如：

```make
CUDA_PATH = /usr/local/cuda
```

H100 使用：

```make
-gencode=arch=compute_90,code=sm_90
```

A100 使用：

```make
-gencode=arch=compute_80,code=sm_80
```

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
