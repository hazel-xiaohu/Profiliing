#ifndef PANGULU_COMMON_H
#define PANGULU_COMMON_H

#define GPU_OPEN

// #define CHECK_ANS
#define KERNEL_TEST // kernels test

// #define CHECK_KERNEL_ERROR//check error

// #define PRINT_TIME

#define KERNEL_PRINT_INFO // output all information

#ifdef KERNEL_PRINT_INFO
#define OUTPUT_FILE_PATH "../test/csv_out"
#endif
#define MULTI_RUN // run multis time

#ifdef KERNEL_TEST

#define SSSSM_CPU_DENSE_TEST
#define SSSSM_CPU_THMKL_TEST
// #define SSSSM_CPU_SPMM_TEST
#define SSSSM_GPU_DENSE_TEST
#define SSSSM_GPU_SPARSE_TEST
#define GETRF_CPU_DENSE_TEST
#define GETRF_CPU_SPARSE_TEST
#define GETRF_GPU_DENSE_TEST
#define GETRF_GPU_SPARSE_TEST
#define GESSM_CPU_1_TEST
#define GESSM_CPU_2_TEST
#define GESSM_CPU_3_TEST
#define GESSM_CPU_4_TEST
#define GESSM_CPU_5_TEST
#define GESSM_CPU_6_TEST
#define GESSM_GPU_7_TEST
#define GESSM_GPU_8_TEST
#define GESSM_GPU_9_TEST
#define GESSM_GPU_10_TEST
#define GESSM_GPU_11_TEST
#define TSTRF_CPU_1_TEST
#define TSTRF_CPU_2_TEST
#define TSTRF_CPU_3_TEST
#define TSTRF_CPU_4_TEST
#define TSTRF_CPU_5_TEST
#define TSTRF_CPU_6_TEST
#define TSTRF_GPU_7_TEST
#define TSTRF_GPU_8_TEST
#define TSTRF_GPU_9_TEST
#define TSTRF_GPU_10_TEST
#define TSTRF_GPU_11_TEST
#endif

#define CPU_OPTION
#define CHECK_TIME

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <omp.h>

typedef int64_t int_t;
typedef int32_t int_32t;
typedef int idx_int;

#define MPI_INT_TYPE MPI_LONG

#define calculate_type double
// #if calculate_type==double
#define MPI_VAL_TYPE MPI_DOUBLE
// #elif calculate_type==float
// #define MPI_VAL_TYPE MPI_FLOAT
// #endif

#define ERROR 0.000001
#define CHECK_ERROR 0.0001
#define BIN_LENGTH 12
#define pangulu_exchange_PQ(row, P) \
    (row + P - 1) % P

#define calculate_offset(offset_init, now_level, PQ_length) \
    ((offset_init - now_level % PQ_length) < 0) ? (offset_init - now_level % PQ_length + PQ_length) : (offset_init - now_level % PQ_length)

#define pangulu_Calculate_Block(nrow, block) \
    nrow / block + ((nrow % block) ? 1 : 0)

#define calculate_diagonal_rank(level, P, Q) \
    level % P *Q + level % Q

extern int_t CPU_MEMORY;
extern int_t GPU_MEMORY;

#ifdef CHECK_TIME

extern double TIME_transport;
extern double TIME_isend;
extern double TIME_receive;
extern double TIME_getrf;
extern double TIME_tstrf;
extern double TIME_gessm;
extern double TIME_gessm_dense;
extern double TIME_gessm_sparse;
extern double TIME_ssssm;
extern double TIME_cuda_memcpy;
extern double TIME_wait;
extern struct timeval GET_TIME_START, GET_TIME_END;
#endif

extern calculate_type *TEMP_A_value;

#ifdef GPU_OPEN
extern calculate_type *CUDA_TEMP_value;
extern idx_int *CUDA_B_idx_COL;
#endif

enum Pangulu_print_kernel_type
{
    SSSSM = 0,
    GETRF,
    GESSM,
    TSTRF
};
enum Pangulu_kernel_type
{
    SSSSM_CPU_DENSE = 0,
    SSSSM_CPU_THMKL,
    SSSSM_CPU_SPMM,
    SSSSM_GPU_DENSE,
    SSSSM_GPU_SPARSE,
    GETRF_CPU_DENSE,
    GETRF_CPU_SPARSE,
    GETRF_GPU_DENSE,
    GETRF_GPU_SPARSE,
    GESSM_CPU_1,
    GESSM_CPU_2,
    GESSM_CPU_3,
    GESSM_CPU_4,
    GESSM_CPU_5,
    GESSM_CPU_6,
    GESSM_GPU_7,
    GESSM_GPU_8,
    GESSM_GPU_9,
    GESSM_GPU_10,
    GESSM_GPU_11,
    TSTRF_CPU_1,
    TSTRF_CPU_2,
    TSTRF_CPU_3,
    TSTRF_CPU_4,
    TSTRF_CPU_5,
    TSTRF_CPU_6,
    TSTRF_GPU_7,
    TSTRF_GPU_8,
    TSTRF_GPU_9,
    TSTRF_GPU_10,
    TSTRF_GPU_11,

};

typedef struct pangulu_kernel_time
{
    double SSSSM_CPU_DENSE_TIME;
    double SSSSM_CPU_THMKL_TIME;
    double SSSSM_CPU_SPMM_TIME;
    double SSSSM_GPU_DENSE_TIME;
    double SSSSM_GPU_SPARSE_TIME;
    double GETRF_CPU_DENSE_TIME;
    double GETRF_CPU_SPARSE_TIME;
    double GETRF_GPU_DENSE_TIME;
    double GETRF_GPU_SPARSE_TIME;
    double GESSM_CPU_1_TIME;
    double GESSM_CPU_2_TIME;
    double GESSM_CPU_3_TIME;
    double GESSM_CPU_4_TIME;
    double GESSM_CPU_5_TIME;
    double GESSM_CPU_6_TIME;
    double GESSM_GPU_7_TIME;
    double GESSM_GPU_8_TIME;
    double GESSM_GPU_9_TIME;
    double GESSM_GPU_10_TIME;
    double GESSM_GPU_11_TIME;
    double TSTRF_CPU_1_TIME;
    double TSTRF_CPU_2_TIME;
    double TSTRF_CPU_3_TIME;
    double TSTRF_CPU_4_TIME;
    double TSTRF_CPU_5_TIME;
    double TSTRF_CPU_6_TIME;
    double TSTRF_GPU_7_TIME;
    double TSTRF_GPU_8_TIME;
    double TSTRF_GPU_9_TIME;
    double TSTRF_GPU_10_TIME;
    double TSTRF_GPU_11_TIME;
} pangulu_kernel_time;

typedef struct pangulu_Smatrix
{

    calculate_type *value;
    calculate_type *value_CSC;
    int_t *CSR_to_CSC_index;
    int_t *CSC_to_CSR_index;
    int_t *rowpointer;
    idx_int *columnindex;
    int_t *columnpointer;
    idx_int *rowindex;
    int_t column;
    int_t row;
    int_t nnz;
    int_32t *nnzU;
    int_t *bin_rowpointer;
    int_t *bin_rowindex;

#ifdef GPU_OPEN
    int_t *CUDA_rowpointer;
    idx_int *CUDA_columnindex;
    calculate_type *CUDA_value;
    int_32t *CUDA_nnzU;
    int_t *CUDA_bin_rowpointer;
    int_t *CUDA_bin_rowindex;

    int *graphInDegree;
    int *d_graphInDegree;
    int *d_id_extractor;
    calculate_type *d_left_sum;
#else
    int_t num_lev;
    int_t *level_idx;
    int_t *level_size;
#endif

} pangulu_Smatrix;

typedef struct pangulu_vector
{
    calculate_type *value;
    int_t row;
} pangulu_vector;

typedef struct matrix_info
{
    int n;
    int nnz;
    int max_row_nnz;
    int min_row_nnz;
    int max_col_nnz;
    int min_col_nnz;
    double average_row_nnz;
    double average_col_nnz;
    double density;
    double variety_col;
    double variety_row;

} matrix_info;

#endif
