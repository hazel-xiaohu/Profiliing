# H100 incremental profiling: block size = 2048 × N, N=1..8

Existing formal sizes are 2048, 4096, 8192 and 16384.  This campaign collects
the missing sizes 6144, 10240, 12288 and 14336 for the frozen 24-matrix list.

Protocol: H100 80GB, one MPI rank, forced dense and sparse SSSSM campaigns,
one measured process per configuration, no warmup, 1800 s timeout, no finite
residual threshold.  Total: 24 × 4 × 2 = 192 measured runs.

The prior missing GESSM/TSTRF matrix was lowThrust_13.  Its old sizes have
already been repaired with 81 logical samples per kernel and incorporated into
the H100 platform model-ready files.  It participates normally in all four new
sizes here.

Run/resume:

```bash
bash experiments/incremental_sizes_n1_n8_h100_v1/run.sh
```

Status:

```bash
python3 experiments/incremental_sizes_n1_n8_h100_v1/status.py
```
