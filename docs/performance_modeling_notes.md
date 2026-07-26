# PanguLU Performance Modeling Notes

This note records the offline profiling changes made for task-level and GPU
interface-level performance modeling.

## Scope

The target setup is single GPU execution:

```text
P = 1
Q = 1
rank = 0
```

The target CUDA kernels for future GPU profiling are:

```text
GETRF  -> pangulu_getrf_cuda_kernel
GESSM  -> pangulu_gessm_fp64_cuda_v7
TSTRF  -> pangulu_tstrf_fp64_cuda_v7
SSSSM  -> pangulu_ssssm_cuda_kernel
```

## Changes Made

### Task Catalog

`src/pangulu_calib.h` adds catalog output controlled by:

```bash
CALIB_MODE=catalog
CALIB_CATALOG_DIR=calib_catalog
```

Each row is one numerical task. The task is identified by:

```text
matrix_id
kernel_type
level
block_row
block_col
update_k
rank
```

The structural features include:

```text
block_size
nnz_input_0
nnz_input_1
nnz_output
flops
estimated_bytes
operational_intensity
row_nnz_cv
col_nnz_cv
structural_factor
structure_hash
```

`time_ms` is kept for backward compatibility. It currently mirrors
`time_total_ms` when the new profiling record is used.

### Interface Profiling Fields

The catalog now also records fields for modeling interface-level cost:

```text
profile_scope
gpu_kernel_name
time_total_ms
time_kernel_ms
time_memcpy_h2d_ms
time_memcpy_d2h_ms
time_memcpy_d2d_ms
time_preprocess_ms
time_overhead_ms
num_kernel_launches
num_memcpy_h2d
num_memcpy_d2h
num_memcpy_d2d
bytes_h2d
bytes_d2h
bytes_d2d
preprocess_required
input_residency_flags
```

Current CPU-only development environment limitation:

```text
time_total_ms is measured by a CPU wall timer.
time_preprocess_ms is measured for the known outer GESSM/TSTRF GPU preparation block.
CUDA kernel/memcpy breakdown fields are set to -1 when not measured.
```

`-1` means unknown. It is intentionally not written as `0`, because `0` would
incorrectly imply the cost was measured and found to be zero.

## Timing Boundaries

### GETRF

Timed region:

```text
pangulu_getrf_interface(...)
```

Recorded target GPU kernel name:

```text
pangulu_getrf_cuda_kernel
```

### TSTRF

Timed region:

```text
optional first-use U GPU preparation
pangulu_tstrf_interface(...)
```

The first-use preparation block is:

```text
pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U)
pangulu_tstrf_preprocess(U)
```

Recorded target GPU kernel name:

```text
pangulu_tstrf_fp64_cuda_v7
```

### GESSM

Timed region:

```text
optional first-use L GPU preparation
pangulu_gessm_interface(...)
```

The first-use preparation block is:

```text
pangulu_Smatrix_CUDA_memcpy_complete_CSC(L, L)
pangulu_gessm_preprocess(L)
```

Recorded target GPU kernel name:

```text
pangulu_gessm_fp64_cuda_v7
```

### SSSSM

Timed region:

```text
pangulu_ssssm_interface(...)
```

Recorded target GPU kernel name:

```text
pangulu_ssssm_cuda_kernel
```

## Baseline Model Script

`script/perf_model_baseline.py` trains one log-linear ridge model per kernel:

```text
log1p(time) = f(task structural features)
```

The script uses `time_total_ms` when available, and falls back to `time_ms` for
older catalogs.

Example:

```bash
python3 script/perf_model_baseline.py --repeats 20 'calib_catalog/*.catalog.csv'
```

## Previously Observed Result

Using the existing `calib_catalog/ACTIVSg2000.rank0.catalog.csv`, the SSSSM
baseline was the most stable:

```text
SSSSM samples: 246
MAPE: about 17%
p50 relative error: about 15%
p90 relative error: about 34%
```

GETRF, GESSM, and TSTRF had fewer samples and higher split sensitivity.

## GPU Platform Next Steps

On a GPU platform, the next useful instrumentation step is to fill the breakdown
fields with real measurements:

```text
time_kernel_ms       via CUDA events around the target CUDA kernel launch
time_memcpy_*_ms     via CUDA events or CPU timers around memcpy wrappers
num_kernel_launches  count launches inside the interface path
num_memcpy_*         count memcpy calls inside the interface path
bytes_*              sum copied bytes by direction
```

Keep `time_total_ms` as the end-to-end label for partition decisions. Use
`time_kernel_ms` mainly for kernel optimization analysis.

## Recommended Profiling Run

Rebuild on the GPU platform, remove old catalogs, and run:

```bash
make -B src
make lib
make test
rm -f calib_catalog/*.catalog.csv
CALIB_MODE=catalog CALIB_CATALOG_DIR=calib_catalog mpirun -np 1 ./test/PanguLU -F ./matrix/<matrix>.mtx
python3 script/perf_model_baseline.py --repeats 20 'calib_catalog/*.catalog.csv'
```

Collect catalogs from multiple matrices before trusting the model for
partitioning decisions.
