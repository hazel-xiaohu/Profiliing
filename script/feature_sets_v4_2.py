#!/usr/bin/env python3
"""Online-safe v4.2 features after removing redundant max/mean ratios and outputs."""
DIMS=["replay_main_dimension"]
BASELINE_FIELDS=["replay_operation_count_v2","replay_base_bytes_est"]
COL_SUFFIXES=("col_nnz_mean","col_nnz_max","col_nnz_cv","col_empty_fraction")
ROW_SUFFIXES=("row_nnz_mean","row_nnz_max","row_nnz_cv","row_empty_fraction")
def input_col_features(count):
 return [f"replay_input_{i}_{suffix}" for i in range(count) for suffix in COL_SUFFIXES]
def input_row_features(count):
 return [f"replay_input_{i}_{suffix}" for i in range(count) for suffix in ROW_SUFFIXES]

# GETRF: A only; L/U output NNZ and A-column max/mean ratio removed.
GETRF_FEATURES=DIMS+["replay_nnz_input_0"]+input_col_features(1)
# GESSM: A,L; X output NNZ and both column max/mean ratios removed.
GESSM_FEATURES=DIMS+["replay_nnz_input_0","replay_nnz_input_1"]+input_col_features(2)
# TSTRF: A,U; X output NNZ and both row max/mean ratios removed.
TSTRF_FEATURES=DIMS+["replay_nnz_input_0","replay_nnz_input_1"]+input_row_features(2)
# Dense SSSSM: A,L,U; all three column max/mean ratios removed.
SSSSM_FEATURES=DIMS+["replay_nnz_input_0","replay_nnz_input_1","replay_nnz_input_2"]+input_col_features(3)
FEATURES_BY_KERNEL={"GETRF":GETRF_FEATURES,"GESSM":GESSM_FEATURES,"TSTRF":TSTRF_FEATURES,
                    "SSSSM_dense":SSSSM_FEATURES,"SSSSM_sparse":SSSSM_FEATURES}
