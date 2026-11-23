#!/usr/bin/env python3
"""Kernel-specific online-safe-v4 feature allowlists.

Derived from the 2026-07-28 ablation and split-use audit.  Analytical
baseline fields and the new rectangular dimensions are retained explicitly.
"""
DIMS = ["replay_main_dimension"]
BASELINE_FIELDS = ["replay_operation_count_v2", "replay_base_bytes_est"]
RESIDUAL_ANALYTICAL_FEATURES = []
COL_SUFFIXES = ("col_nnz_mean", "col_nnz_max", "col_nnz_cv", "col_nnz_max_mean_ratio", "col_empty_fraction")
def input_col_features(count):
    return [f"replay_input_{i}_{suffix}" for i in range(count) for suffix in COL_SUFFIXES]
ROW_SUFFIXES = ("row_nnz_mean", "row_nnz_max", "row_nnz_cv", "row_nnz_max_mean_ratio", "row_empty_fraction")
def input_row_features(indices):
    return [f"replay_input_{i}_{suffix}" for i in indices for suffix in ROW_SUFFIXES]
GETRF_FEATURES = DIMS + [
 "replay_nnz_input_0","replay_nnz_output_0","replay_nnz_output_1",
] + input_col_features(1) + RESIDUAL_ANALYTICAL_FEATURES
GT_FEATURES = DIMS + [
 "replay_nnz_input_0","replay_nnz_input_1","replay_nnz_output_0",
 "replay_derived_csr_index_gap_mean","replay_derived_csr_index_gap_cv",
 "replay_derived_csr_consecutive_fraction",
] + input_col_features(2) + RESIDUAL_ANALYTICAL_FEATURES
GESSM_FEATURES = [f for f in GT_FEATURES if not f.startswith("replay_derived_csr_")]
TSTRF_FEATURES = [f for f in GT_FEATURES if not f.startswith("replay_derived_csr_") and not f.startswith("replay_input_")] + input_row_features((0, 1))
SSSSM_FEATURES = DIMS + [
 "replay_nnz_input_0","replay_nnz_input_1","replay_nnz_input_2",
 ] + input_col_features(3) + RESIDUAL_ANALYTICAL_FEATURES
FEATURES_BY_KERNEL={"GETRF":GETRF_FEATURES,"GESSM":GESSM_FEATURES,"TSTRF":TSTRF_FEATURES,
                    "SSSSM_dense":SSSSM_FEATURES,"SSSSM_sparse":SSSSM_FEATURES}
