#!/usr/bin/env python3
"""Kernel-specific online-safe-v3 feature allowlists.

Derived from the 2026-07-28 ablation and split-use audit.  Analytical
baseline fields and the new rectangular dimensions are retained explicitly.
"""
DIMS = ["replay_main_dimension"]
BASELINE_FIELDS = ["replay_operation_count_v2", "replay_base_bytes_est"]
RESIDUAL_ANALYTICAL_FEATURES = []
GETRF_FEATURES = DIMS + [
 "replay_nnz_input_0","replay_nnz_output_0","replay_nnz_output_1",
 "replay_row_nnz_mean","replay_row_nnz_max","replay_row_nnz_cv",
 "replay_row_nnz_max_mean_ratio","replay_row_empty_fraction",
 "replay_index_value_byte_ratio","replay_reuse_factor_est",
 "replay_block_csc_index_gap_mean","replay_block_csc_index_gap_cv",
 "replay_block_csc_consecutive_fraction",
] + RESIDUAL_ANALYTICAL_FEATURES
GT_FEATURES = DIMS + [
 "replay_nnz_input_0","replay_nnz_input_1","replay_nnz_output_0",
 "replay_row_nnz_mean","replay_row_nnz_max","replay_row_nnz_cv",
 "replay_row_nnz_max_mean_ratio","replay_row_empty_fraction",
 "replay_col_nnz_max","replay_col_nnz_cv","replay_col_nnz_max_mean_ratio",
 "replay_col_empty_fraction",
 "replay_index_value_byte_ratio","replay_reuse_factor_est",
 "replay_block_csc_index_gap_mean","replay_block_csc_index_gap_cv",
 "replay_block_csc_consecutive_fraction",
 "replay_derived_csr_index_gap_mean","replay_derived_csr_index_gap_cv",
 "replay_derived_csr_consecutive_fraction",
] + RESIDUAL_ANALYTICAL_FEATURES
GESSM_FEATURES = [f for f in GT_FEATURES if not f.startswith("replay_derived_csr_")]
TSTRF_FEATURES = [f for f in GT_FEATURES if not f.startswith("replay_block_csc_")]
SSSSM_FEATURES = DIMS + [
 "replay_nnz_input_0","replay_nnz_input_1","replay_nnz_input_2",
 "replay_row_nnz_mean","replay_row_nnz_max",
 "replay_row_nnz_max_mean_ratio","replay_row_empty_fraction",
 "replay_col_nnz_max","replay_col_nnz_cv","replay_col_nnz_max_mean_ratio",
 "replay_col_empty_fraction",
 "replay_value_bytes_est","replay_index_value_byte_ratio","replay_reuse_factor_est",
 "replay_block_csc_index_gap_mean","replay_block_csc_index_gap_cv",
 "replay_block_csc_consecutive_fraction",
] + RESIDUAL_ANALYTICAL_FEATURES
FEATURES_BY_KERNEL={"GETRF":GETRF_FEATURES,"GESSM":GESSM_FEATURES,"TSTRF":TSTRF_FEATURES,
                    "SSSSM_dense":SSSSM_FEATURES,"SSSSM_sparse":SSSSM_FEATURES}
