#ifndef PANGULU_CALIB_H
#define PANGULU_CALIB_H

#include "pangulu_common.h"

#include <errno.h>
#include <stdint.h>
#include <sys/stat.h>

typedef enum pangulu_calib_kernel
{
    PANGULU_CALIB_GETRF,
    PANGULU_CALIB_GESSM,
    PANGULU_CALIB_TSTRF,
    PANGULU_CALIB_SSSSM
} pangulu_calib_kernel;

typedef struct pangulu_calib_profile
{
    const char *profile_scope;
    const char *gpu_kernel_name;
    double time_total_ms;
    double time_kernel_ms;
    double time_memcpy_h2d_ms;
    double time_memcpy_d2h_ms;
    double time_memcpy_d2d_ms;
    double time_preprocess_ms;
    double time_overhead_ms;
    int num_kernel_launches;
    int num_memcpy_h2d;
    int num_memcpy_d2h;
    int num_memcpy_d2d;
    int_t bytes_h2d;
    int_t bytes_d2h;
    int_t bytes_d2d;
    int preprocess_required;
    int input_residency_flags;
} pangulu_calib_profile;

/* Captured before the target kernel runs; contains no runtime measurements. */
typedef struct pangulu_calib_task_features
{
    int_t nnz_input_0, nnz_input_1, nnz_input_2;
    int_t nnz_output_0, nnz_output_1;
    int_t flops, estimated_bytes;
    double operational_intensity;
    double row_nnz_cv, col_nnz_cv, structural_factor;
    uint64_t structure_hash;
    int_t block_rows, block_cols, min_dimension, max_dimension, main_dimension;
    double aspect_ratio;
    double density_input_0, density_input_1, density_input_2;
    int_t num_macs, num_divisions, operation_count_v2;
    double row_nnz_mean, row_nnz_max, row_nnz_max_mean_ratio, row_empty_fraction;
    double col_nnz_mean, col_nnz_max, col_nnz_max_mean_ratio, col_empty_fraction;
    double input_col_nnz_mean[3], input_col_nnz_max[3], input_col_nnz_cv[3];
    double input_col_nnz_max_mean_ratio[3], input_col_empty_fraction[3];
    double input_row_nnz_mean[3], input_row_nnz_max[3], input_row_nnz_cv[3];
    double input_row_nnz_max_mean_ratio[3], input_row_empty_fraction[3];
    double work_mean, work_max, work_cv, work_max_mean_ratio, active_work_fraction;
    int_t value_bytes_est, index_bytes_est, pointer_bytes_est;
    int_t output_write_bytes_est, dense_workspace_bytes_est, base_bytes_est;
    double index_value_byte_ratio, reuse_factor_est;
    double block_csc_index_gap_mean, block_csc_index_gap_cv, block_csc_consecutive_fraction;
    double derived_csr_index_gap_mean, derived_csr_index_gap_cv, derived_csr_consecutive_fraction;
    int_t active_rows, active_cols;
    uint64_t canonical_structure_hash_v2;
    int_t ssssm_bins[10], ssssm_nonempty_bins, ssssm_sparse_launches_est;
    double ssssm_max_bin_fraction;
} pangulu_calib_task_features;

#ifdef GPU_OPEN
typedef struct pangulu_calib_cuda_event
{
    cudaEvent_t start;
    cudaEvent_t stop;
    int enabled;
} pangulu_calib_cuda_event;

static thread_local pangulu_calib_profile *pangulu_calib_active_profile = NULL;

static void pangulu_calib_profile_activate(pangulu_calib_profile *profile)
{
    const char *mode = getenv("CALIB_MODE");
    if (mode != NULL && strcmp(mode, "catalog") == 0)
    {
        pangulu_calib_active_profile = profile;
        profile->time_kernel_ms = 0.0;
        profile->num_kernel_launches = 0;
    }
    else
    {
        pangulu_calib_active_profile = NULL;
    }
}

static void pangulu_calib_profile_deactivate(void)
{
    pangulu_calib_active_profile = NULL;
}

static void pangulu_calib_kernel_event_begin(pangulu_calib_cuda_event *event)
{
    event->enabled = 0;
    if (pangulu_calib_active_profile == NULL)
    {
        return;
    }
    if (cudaEventCreate(&event->start) != cudaSuccess)
    {
        return;
    }
    if (cudaEventCreate(&event->stop) != cudaSuccess)
    {
        cudaEventDestroy(event->start);
        return;
    }
    if (cudaEventRecord(event->start, 0) != cudaSuccess)
    {
        cudaEventDestroy(event->start);
        cudaEventDestroy(event->stop);
        return;
    }
    event->enabled = 1;
}

static void pangulu_calib_kernel_event_end(pangulu_calib_cuda_event *event, int kernel_launches)
{
    if (!event->enabled)
    {
        return;
    }
    float elapsed_ms = 0.0f;
    cudaError_t status = cudaEventRecord(event->stop, 0);
    if (status == cudaSuccess)
    {
        status = cudaEventSynchronize(event->stop);
    }
    if (status == cudaSuccess)
    {
        status = cudaEventElapsedTime(&elapsed_ms, event->start, event->stop);
    }
    if (status == cudaSuccess && pangulu_calib_active_profile != NULL)
    {
        pangulu_calib_active_profile->time_kernel_ms += (double)elapsed_ms;
        pangulu_calib_active_profile->num_kernel_launches += kernel_launches;
    }
    cudaEventDestroy(event->start);
    cudaEventDestroy(event->stop);
    event->enabled = 0;
}
#else
static void pangulu_calib_profile_activate(pangulu_calib_profile *profile) { (void)profile; }
static void pangulu_calib_profile_deactivate(void) {}
#endif

static FILE *pangulu_calib_catalog_fps[4] = {NULL, NULL, NULL, NULL};
static int pangulu_calib_catalog_enabled = -1;
static char pangulu_calib_matrix_id[256] = "unknown";

static uint64_t pangulu_calib_hash_u64(uint64_t hash, uint64_t value)
{
    hash ^= value;
    hash *= 1099511628211ULL;
    return hash;
}

static const char *pangulu_calib_kernel_name(pangulu_calib_kernel kernel)
{
    switch (kernel)
    {
    case PANGULU_CALIB_GETRF:
        return "GETRF";
    case PANGULU_CALIB_GESSM:
        return "GESSM";
    case PANGULU_CALIB_TSTRF:
        return "TSTRF";
    case PANGULU_CALIB_SSSSM:
        return "SSSSM";
    default:
        return "UNKNOWN";
    }
}

static int pangulu_calib_ssssm_use_dense_kernel(pangulu_Smatrix *A)
{
    const char *mode = getenv("PANGULU_SSSSM_KERNEL");
    if (mode != NULL && strcmp(mode, "sparse") == 0) return 0;
    if (mode != NULL && strcmp(mode, "dense") == 0) return 1;
    if (A == NULL || A->columnpointer == NULL || A->row <= 0) return 0;
    const double n = (double)A->row;
    const int_t nnz = A->columnpointer[A->row] - A->columnpointer[0];
    return (double)nnz / (n * n) >= 0.001;
}

static const char *pangulu_calib_target_gpu_kernel_name(pangulu_calib_kernel kernel)
{
    switch (kernel)
    {
    case PANGULU_CALIB_GETRF:
        return "pangulu_getrf_cuda_kernel";
    case PANGULU_CALIB_GESSM:
        return "pangulu_gessm_fp64_cuda_v7";
    case PANGULU_CALIB_TSTRF:
        return "pangulu_tstrf_fp64_cuda_v7";
    case PANGULU_CALIB_SSSSM:
        return "pangulu_ssssm_cuda_kernel";
    default:
        return "unknown";
    }
}

static pangulu_calib_profile pangulu_calib_profile_unknown(pangulu_calib_kernel kernel)
{
    pangulu_calib_profile profile;
    profile.profile_scope = "gpu_task_interface";
    profile.gpu_kernel_name = pangulu_calib_target_gpu_kernel_name(kernel);
    profile.time_total_ms = -1.0;
    profile.time_kernel_ms = -1.0;
    profile.time_memcpy_h2d_ms = -1.0;
    profile.time_memcpy_d2h_ms = -1.0;
    profile.time_memcpy_d2d_ms = -1.0;
    profile.time_preprocess_ms = -1.0;
    profile.time_overhead_ms = -1.0;
    profile.num_kernel_launches = -1;
    profile.num_memcpy_h2d = -1;
    profile.num_memcpy_d2h = -1;
    profile.num_memcpy_d2d = -1;
    profile.bytes_h2d = -1;
    profile.bytes_d2h = -1;
    profile.bytes_d2d = -1;
    profile.preprocess_required = -1;
    profile.input_residency_flags = -1;
    return profile;
}

static void pangulu_calib_make_matrix_id(const char *file_name)
{
    const char *base = file_name;
    if (base == NULL || base[0] == '\0')
    {
        strcpy(pangulu_calib_matrix_id, "unknown");
        return;
    }

    const char *slash = strrchr(base, '/');
    if (slash != NULL)
    {
        base = slash + 1;
    }

    size_t out = 0;
    for (size_t i = 0; base[i] != '\0' && out + 1 < sizeof(pangulu_calib_matrix_id); ++i)
    {
        char c = base[i];
        if (c == '.' && strcmp(base + i, ".mtx") == 0)
        {
            break;
        }
        pangulu_calib_matrix_id[out++] = ((c == ',') || (c == ' ') || (c == '\t')) ? '_' : c;
    }
    pangulu_calib_matrix_id[out] = '\0';
    if (out == 0)
    {
        strcpy(pangulu_calib_matrix_id, "unknown");
    }
}

static int pangulu_calib_is_feature_replay(void)
{
    const char *mode = getenv("CALIB_MODE");
    return mode != NULL && strcmp(mode, "feature_replay") == 0;
}

static int pangulu_calib_is_catalog_enabled(void)
{
    if (pangulu_calib_catalog_enabled >= 0)
    {
        return pangulu_calib_catalog_enabled;
    }

    const char *mode = getenv("CALIB_MODE");
    pangulu_calib_catalog_enabled =
        (mode != NULL && (strcmp(mode, "catalog") == 0 ||
                          strcmp(mode, "feature_replay") == 0)) ? 1 : 0;
    return pangulu_calib_catalog_enabled;
}

static void pangulu_calib_write_catalog_header(FILE *fp)
{
    fprintf(fp,
            "task_id,matrix_id,kernel_type,kernel_variant,level,block_row,block_col,update_k,rank,block_size,"
            "process_rows,process_cols,"
            "nnz_input_0,nnz_input_1,nnz_input_2,nnz_output_0,nnz_output_1,"
            "flops,estimated_bytes,operational_intensity,"
            "row_nnz_cv,col_nnz_cv,structural_factor,structure_hash,"
            "feature_schema_version,block_rows,block_cols,aspect_ratio,min_dimension,max_dimension,main_dimension,density_input_0,density_input_1,density_input_2,"
            "num_macs,num_divisions,operation_count_v2,row_nnz_mean,row_nnz_max,row_nnz_max_mean_ratio,row_empty_fraction,"
            "col_nnz_mean,col_nnz_max,col_nnz_max_mean_ratio,col_empty_fraction,"
            "input_0_col_nnz_mean,input_0_col_nnz_max,input_0_col_nnz_cv,input_0_col_nnz_max_mean_ratio,input_0_col_empty_fraction,"
            "input_1_col_nnz_mean,input_1_col_nnz_max,input_1_col_nnz_cv,input_1_col_nnz_max_mean_ratio,input_1_col_empty_fraction,"
            "input_2_col_nnz_mean,input_2_col_nnz_max,input_2_col_nnz_cv,input_2_col_nnz_max_mean_ratio,input_2_col_empty_fraction,"
            "input_0_row_nnz_mean,input_0_row_nnz_max,input_0_row_nnz_cv,input_0_row_nnz_max_mean_ratio,input_0_row_empty_fraction,"
            "input_1_row_nnz_mean,input_1_row_nnz_max,input_1_row_nnz_cv,input_1_row_nnz_max_mean_ratio,input_1_row_empty_fraction,"
            "input_2_row_nnz_mean,input_2_row_nnz_max,input_2_row_nnz_cv,input_2_row_nnz_max_mean_ratio,input_2_row_empty_fraction,"
            "work_mean,work_max,work_cv,work_max_mean_ratio,active_work_fraction,"
            "value_bytes_est,index_bytes_est,pointer_bytes_est,output_write_bytes_est,dense_workspace_bytes_est,base_bytes_est,"
            "index_value_byte_ratio,reuse_factor_est,block_csc_index_gap_mean,block_csc_index_gap_cv,block_csc_consecutive_fraction,derived_csr_index_gap_mean,derived_csr_index_gap_cv,derived_csr_consecutive_fraction,"
            "active_rows,active_cols,canonical_structure_hash_v2,"
            "ssssm_bin_1,ssssm_bin_2,ssssm_bin_3,ssssm_bin_4,ssssm_bin_5,ssssm_bin_6,ssssm_bin_7,ssssm_bin_8,ssssm_bin_9,ssssm_bin_10,"
            "ssssm_nonempty_bins,ssssm_max_bin_fraction,ssssm_sparse_launches_est,time_ms,"
            "profile_scope,gpu_kernel_name,time_total_ms,time_kernel_ms,time_memcpy_h2d_ms,"
            "time_memcpy_d2h_ms,time_memcpy_d2d_ms,time_preprocess_ms,time_overhead_ms,"
            "num_kernel_launches,num_memcpy_h2d,num_memcpy_d2h,num_memcpy_d2d,"
            "bytes_h2d,bytes_d2h,bytes_d2d,preprocess_required,input_residency_flags\n");
}

static void pangulu_calib_open_catalogs(int_32t rank)
{
    if (!pangulu_calib_is_catalog_enabled())
    {
        return;
    }

    for (int kernel = PANGULU_CALIB_GETRF; kernel <= PANGULU_CALIB_SSSSM; ++kernel)
    {
        if (pangulu_calib_catalog_fps[kernel] != NULL)
        {
            return;
        }
    }

    const char *dir = getenv("CALIB_CATALOG_DIR");
    if (dir == NULL || dir[0] == 0)
    {
        dir = "calib_catalog";
    }

    if (mkdir(dir, 0755) != 0 && errno != EEXIST)
    {
        fprintf(stderr, "PanguLU catalog: failed to create %s\n", dir);
        pangulu_calib_catalog_enabled = 0;
        return;
    }

    pangulu_calib_make_matrix_id(block_matrix_name);

    for (int kernel = PANGULU_CALIB_GETRF; kernel <= PANGULU_CALIB_SSSSM; ++kernel)
    {
        char path[1024];
        snprintf(path,
                 sizeof(path),
                 "%s/%s.rank%d.%s.catalog.csv",
                 dir,
                 pangulu_calib_matrix_id,
                 rank,
                 pangulu_calib_kernel_name((pangulu_calib_kernel)kernel));
        pangulu_calib_catalog_fps[kernel] = fopen(path, "w");
        if (pangulu_calib_catalog_fps[kernel] == NULL)
        {
            fprintf(stderr, "PanguLU catalog: failed to open %s\n", path);
            pangulu_calib_catalog_enabled = 0;
            for (int opened = PANGULU_CALIB_GETRF; opened <= PANGULU_CALIB_SSSSM; ++opened)
            {
                if (pangulu_calib_catalog_fps[opened] != NULL)
                {
                    fclose(pangulu_calib_catalog_fps[opened]);
                    pangulu_calib_catalog_fps[opened] = NULL;
                }
            }
            return;
        }
        pangulu_calib_write_catalog_header(pangulu_calib_catalog_fps[kernel]);
    }
}

static void pangulu_calib_close_catalog(void)
{
    for (int kernel = PANGULU_CALIB_GETRF; kernel <= PANGULU_CALIB_SSSSM; ++kernel)
    {
        if (pangulu_calib_catalog_fps[kernel] != NULL)
        {
            fclose(pangulu_calib_catalog_fps[kernel]);
            pangulu_calib_catalog_fps[kernel] = NULL;
        }
    }
}

static double pangulu_calib_cv_from_ptr(int_t *ptr, int_t length)
{
    if (ptr == NULL || length <= 0)
    {
        return 0.0;
    }

    double sum = 0.0;
    for (int_t i = 0; i < length; ++i)
    {
        sum += (double)(ptr[i + 1] - ptr[i]);
    }
    double mean = sum / (double)length;
    if (mean <= 0.0)
    {
        return 0.0;
    }

    double var = 0.0;
    for (int_t i = 0; i < length; ++i)
    {
        double diff = (double)(ptr[i + 1] - ptr[i]) - mean;
        var += diff * diff;
    }
    return sqrt(var / (double)length) / mean;
}

static uint64_t pangulu_calib_hash_ptr(uint64_t hash, int_t *ptr, int_t length)
{
    if (ptr == NULL || length <= 0)
    {
        return pangulu_calib_hash_u64(hash, 0);
    }

    for (int_t i = 0; i < length; ++i)
    {
        hash = pangulu_calib_hash_u64(hash, (uint64_t)(ptr[i + 1] - ptr[i]));
    }
    return hash;
}

static uint64_t pangulu_calib_structure_hash(pangulu_Smatrix **matrices, int count)
{
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < count; ++i)
    {
        pangulu_Smatrix *S = matrices[i];
        if (S == NULL)
        {
            hash = pangulu_calib_hash_u64(hash, 0);
            continue;
        }
        hash = pangulu_calib_hash_u64(hash, (uint64_t)S->row);
        hash = pangulu_calib_hash_u64(hash, (uint64_t)S->column);
        hash = pangulu_calib_hash_u64(hash, (uint64_t)S->nnz);
        hash = pangulu_calib_hash_ptr(hash, S->rowpointer, S->row);
        hash = pangulu_calib_hash_ptr(hash, S->columnpointer, S->column);
    }
    return hash;
}

static int_t pangulu_calib_macs_lu(pangulu_Smatrix *L, pangulu_Smatrix *U)
{
    if (L == NULL || U == NULL || L->columnpointer == NULL || U->rowindex == NULL)
    {
        return 0;
    }

    int_t macs = 0;
    for (int_t i = 0; i < U->nnz; ++i)
    {
        int_t row = U->rowindex[i];
        if (row >= 0 && row < L->column)
        {
            macs += L->columnpointer[row + 1] - L->columnpointer[row];
        }
    }
    return macs;
}

static int_t pangulu_calib_macs_axu(pangulu_Smatrix *A, pangulu_Smatrix *U)
{
    if (A == NULL || U == NULL || A->columnindex == NULL || U->rowpointer == NULL)
    {
        return 0;
    }

    int_t macs = 0;
    for (int_t i = 0; i < A->nnz; ++i)
    {
        int_t col = A->columnindex[i];
        if (col >= 0 && col < U->row)
        {
            macs += U->rowpointer[col + 1] - U->rowpointer[col];
        }
    }
    return macs;
}

static int_t pangulu_calib_macs_lxa(pangulu_Smatrix *A, pangulu_Smatrix *L)
{
    if (A == NULL || L == NULL || A->rowindex == NULL || L->columnpointer == NULL)
    {
        return 0;
    }

    int_t macs = 0;
    for (int_t i = 0; i < A->nnz; ++i)
    {
        int_t row = A->rowindex[i];
        if (row >= 0 && row < L->column)
        {
            macs += L->columnpointer[row + 1] - L->columnpointer[row];
        }
    }
    return macs;
}

typedef struct pangulu_calib_distribution
{
    double mean, max_value, cv, max_mean_ratio, empty_fraction;
    int_t active;
} pangulu_calib_distribution;

static pangulu_calib_distribution pangulu_calib_distribution_from_ptr(int_t *ptr, int_t length)
{
    pangulu_calib_distribution d = {0, 0, 0, 0, 0, 0};
    if (ptr == NULL || length <= 0) return d;
    double sum = 0.0, sumsq = 0.0;
    int_t empty = 0;
    for (int_t i = 0; i < length; ++i)
    {
        double x = (double)(ptr[i + 1] - ptr[i]);
        sum += x; sumsq += x * x;
        if (x > d.max_value) d.max_value = x;
        if (x > 0) d.active++; else empty++;
    }
    d.mean = sum / (double)length;
    if (d.mean > 0)
    {
        double var = sumsq / (double)length - d.mean * d.mean;
        d.cv = sqrt(var > 0 ? var : 0) / d.mean;
        d.max_mean_ratio = d.max_value / d.mean;
    }
    d.empty_fraction = (double)empty / (double)length;
    return d;
}

static double pangulu_calib_density(pangulu_Smatrix *S)
{
    if (S == NULL || S->row <= 0 || S->column <= 0) return 0.0;
    return (double)S->nnz / ((double)S->row * (double)S->column);
}

static void pangulu_calib_index_locality(idx_int *idx, int_t nnz, double *mean, double *cv, double *consecutive)
{
    *mean = *cv = *consecutive = 0.0;
    if (idx == NULL || nnz <= 1) return;
    double sum = 0.0, sumsq = 0.0; int_t cons = 0, count = nnz - 1;
    for (int_t i = 1; i < nnz; ++i)
    {
        double gap = fabs((double)idx[i] - (double)idx[i - 1]);
        sum += gap; sumsq += gap * gap; if (gap <= 1.0) cons++;
    }
    *mean = sum / (double)count;
    if (*mean > 0)
    {
        double var = sumsq / (double)count - (*mean) * (*mean);
        *cv = sqrt(var > 0 ? var : 0) / (*mean);
    }
    *consecutive = (double)cons / (double)count;
}

static uint64_t pangulu_calib_canonical_input_hash(pangulu_calib_kernel kernel,
                                                    pangulu_Smatrix *a,
                                                    pangulu_Smatrix *b,
                                                    pangulu_Smatrix *c,
                                                    pangulu_Smatrix *out0,
                                                    pangulu_Smatrix *out1)
{
    pangulu_Smatrix *matrices[3] = {a, b, c}; int count = 3;
    if (kernel == PANGULU_CALIB_GETRF)
    {
        matrices[0] = a; matrices[1] = out0; matrices[2] = out1;
    }
    return pangulu_calib_structure_hash(matrices, count);
}

static pangulu_calib_task_features pangulu_calib_capture_task_features(
    pangulu_calib_kernel kernel, pangulu_Smatrix *input_0,
    pangulu_Smatrix *input_1, pangulu_Smatrix *input_2,
    pangulu_Smatrix *output_0, pangulu_Smatrix *output_1)
{
    pangulu_calib_task_features f;
    memset(&f, 0, sizeof(f));
    int_t macs = 0;
    if (kernel == PANGULU_CALIB_GETRF)
    {
        macs = pangulu_calib_macs_lu(output_0, output_1);
        f.row_nnz_cv = pangulu_calib_cv_from_ptr(input_0 ? input_0->rowpointer : NULL, input_0 ? input_0->row : 0);
        f.col_nnz_cv = pangulu_calib_cv_from_ptr(input_0 ? input_0->columnpointer : NULL, input_0 ? input_0->column : 0);
        f.structural_factor = PANGULU_MAX(f.row_nnz_cv, f.col_nnz_cv);
    }
    else if (kernel == PANGULU_CALIB_TSTRF)
    {
        macs = pangulu_calib_macs_axu(input_0, input_1);
        f.row_nnz_cv = pangulu_calib_cv_from_ptr(input_1 ? input_1->rowpointer : NULL, input_1 ? input_1->row : 0);
        f.col_nnz_cv = pangulu_calib_cv_from_ptr(input_0 ? input_0->columnpointer : NULL, input_0 ? input_0->column : 0);
        f.structural_factor = f.row_nnz_cv;
    }
    else if (kernel == PANGULU_CALIB_GESSM)
    {
        macs = pangulu_calib_macs_lxa(input_0, input_1);
        f.row_nnz_cv = pangulu_calib_cv_from_ptr(input_0 ? input_0->rowpointer : NULL, input_0 ? input_0->row : 0);
        f.col_nnz_cv = pangulu_calib_cv_from_ptr(input_1 ? input_1->columnpointer : NULL, input_1 ? input_1->column : 0);
        f.structural_factor = f.col_nnz_cv;
    }
    else
    {
        macs = pangulu_calib_macs_lu(input_1, input_2);
        f.row_nnz_cv = pangulu_calib_cv_from_ptr(input_2 ? input_2->rowpointer : NULL, input_2 ? input_2->row : 0);
        f.col_nnz_cv = pangulu_calib_cv_from_ptr(input_1 ? input_1->columnpointer : NULL, input_1 ? input_1->column : 0);
        f.structural_factor = (output_0 && output_0->nnz > 0) ? (double)macs / (double)output_0->nnz : 0.0;
    }
    f.nnz_input_0 = input_0 ? input_0->nnz : 0;
    f.nnz_input_1 = input_1 ? input_1->nnz : 0;
    f.nnz_input_2 = input_2 ? input_2->nnz : 0;
    f.nnz_output_0 = output_0 ? output_0->nnz : 0;
    f.nnz_output_1 = output_1 ? output_1->nnz : 0;
    f.flops = macs * 2;
    f.estimated_bytes = (f.nnz_input_0 + f.nnz_input_1 + f.nnz_input_2 + f.nnz_output_0 + f.nnz_output_1) * (int_t)(sizeof(calculate_type) + sizeof(idx_int));
    f.operational_intensity = f.estimated_bytes > 0 ? (double)f.flops / (double)f.estimated_bytes : 0.0;
    pangulu_Smatrix *matrices[5] = {input_0, input_1, input_2, output_0, output_1};
    f.structure_hash = pangulu_calib_structure_hash(matrices, 5);

    f.block_rows = input_0 ? input_0->row : 0;
    f.block_cols = input_0 ? input_0->column : 0;
    f.aspect_ratio = f.block_cols > 0 ? (double)f.block_rows / (double)f.block_cols : 0.0;
    f.min_dimension = f.block_rows < f.block_cols ? f.block_rows : f.block_cols;
    f.max_dimension = f.block_rows > f.block_cols ? f.block_rows : f.block_cols;
    /* Actual wrappers: GESSM/SSSSM use A->column; GETRF/TSTRF use A->row. */
    f.main_dimension = (kernel == PANGULU_CALIB_GESSM || kernel == PANGULU_CALIB_SSSSM)
                           ? f.block_cols : f.block_rows;
    f.density_input_0 = pangulu_calib_density(input_0);
    f.density_input_1 = pangulu_calib_density(input_1);
    f.density_input_2 = pangulu_calib_density(input_2);
    pangulu_calib_distribution rows = pangulu_calib_distribution_from_ptr(
        input_0 ? input_0->rowpointer : NULL, input_0 ? input_0->row : 0);
    pangulu_calib_distribution cols = pangulu_calib_distribution_from_ptr(
        input_0 ? input_0->columnpointer : NULL, input_0 ? input_0->column : 0);
    f.row_nnz_mean = rows.mean; f.row_nnz_max = rows.max_value;
    f.row_nnz_max_mean_ratio = rows.max_mean_ratio; f.row_empty_fraction = rows.empty_fraction;
    f.col_nnz_mean = cols.mean; f.col_nnz_max = cols.max_value;
    f.col_nnz_max_mean_ratio = cols.max_mean_ratio; f.col_empty_fraction = cols.empty_fraction;
    pangulu_Smatrix *column_inputs[3] = {input_0, input_1, input_2};
    for (int input_index = 0; input_index < 3; ++input_index)
    {
        pangulu_Smatrix *matrix = column_inputs[input_index];
        pangulu_calib_distribution input_cols = pangulu_calib_distribution_from_ptr(
            matrix ? matrix->columnpointer : NULL, matrix ? matrix->column : 0);
        f.input_col_nnz_mean[input_index] = input_cols.mean;
        f.input_col_nnz_max[input_index] = input_cols.max_value;
        f.input_col_nnz_cv[input_index] = input_cols.cv;
        f.input_col_nnz_max_mean_ratio[input_index] = input_cols.max_mean_ratio;
        f.input_col_empty_fraction[input_index] = input_cols.empty_fraction;
        pangulu_calib_distribution input_rows = pangulu_calib_distribution_from_ptr(
            matrix ? matrix->rowpointer : NULL, matrix ? matrix->row : 0);
        f.input_row_nnz_mean[input_index] = input_rows.mean;
        f.input_row_nnz_max[input_index] = input_rows.max_value;
        f.input_row_nnz_cv[input_index] = input_rows.cv;
        f.input_row_nnz_max_mean_ratio[input_index] = input_rows.max_mean_ratio;
        f.input_row_empty_fraction[input_index] = input_rows.empty_fraction;
    }
    f.active_rows = rows.active; f.active_cols = cols.active;

    double work_sum = 0.0, work_sumsq = 0.0, work_max = 0.0;
    int_t work_items = 0, active_work = 0, divisions = 0;
#define PANGULU_CALIB_ADD_WORK(value) do { \
        double calib_w = (double)(value); work_sum += calib_w; work_sumsq += calib_w * calib_w; \
        if (calib_w > work_max) work_max = calib_w; if (calib_w > 0) active_work++; work_items++; \
    } while (0)
    if (kernel == PANGULU_CALIB_GETRF && output_0 && output_1 &&
        output_0->columnpointer && output_1->rowpointer)
    {
        int_t n = PANGULU_MIN(output_0->column, output_1->row);
        f.num_macs = 0;
        for (int_t k = 0; k < n; ++k)
        {
            int_t l = output_0->columnpointer[k + 1] - output_0->columnpointer[k] - 1;
            int_t u = output_1->rowpointer[k + 1] - output_1->rowpointer[k] - 1;
            if (l < 0) l = 0; if (u < 0) u = 0;
            int_t w = l * u; f.num_macs += w; divisions += l; PANGULU_CALIB_ADD_WORK(w);
        }
    }
    else if (kernel == PANGULU_CALIB_TSTRF && input_0 && input_1 &&
             input_0->columnindex && input_1->rowpointer)
    {
        f.num_macs = 0; divisions = input_0->nnz;
        for (int_t i = 0; i < input_0->nnz; ++i)
        {
            int_t k = input_0->columnindex[i], w = 0;
            if (k >= 0 && k < input_1->row) w = input_1->rowpointer[k + 1] - input_1->rowpointer[k] - 1;
            if (w < 0) w = 0; f.num_macs += w; PANGULU_CALIB_ADD_WORK(w);
        }
    }
    else if (kernel == PANGULU_CALIB_GESSM && input_0 && input_1 &&
             input_0->rowindex && input_1->columnpointer)
    {
        f.num_macs = 0;
        for (int_t i = 0; i < input_0->nnz; ++i)
        {
            int_t k = input_0->rowindex[i], w = 0;
            if (k >= 0 && k < input_1->column) w = input_1->columnpointer[k + 1] - input_1->columnpointer[k] - 1;
            if (w < 0) w = 0; f.num_macs += w; PANGULU_CALIB_ADD_WORK(w);
        }
    }
    else if (kernel == PANGULU_CALIB_SSSSM && input_1 && input_2 &&
             input_1->columnpointer && input_2->rowindex)
    {
        f.num_macs = 0;
        for (int_t i = 0; i < input_2->nnz; ++i)
        {
            int_t k = input_2->rowindex[i], w = 0;
            if (k >= 0 && k < input_1->column) w = input_1->columnpointer[k + 1] - input_1->columnpointer[k];
            if (w < 0) w = 0; f.num_macs += w; PANGULU_CALIB_ADD_WORK(w);
        }
    }
#undef PANGULU_CALIB_ADD_WORK
    f.num_divisions = divisions;
    f.operation_count_v2 = 2 * f.num_macs + f.num_divisions;
    if (work_items > 0)
    {
        f.work_mean = work_sum / (double)work_items; f.work_max = work_max;
        double var = work_sumsq / (double)work_items - f.work_mean * f.work_mean;
        f.work_cv = f.work_mean > 0 ? sqrt(var > 0 ? var : 0) / f.work_mean : 0.0;
        f.work_max_mean_ratio = f.work_mean > 0 ? work_max / f.work_mean : 0.0;
        f.active_work_fraction = (double)active_work / (double)work_items;
    }
    int_t input_nnz = f.nnz_input_0 + f.nnz_input_1 + f.nnz_input_2;
    f.value_bytes_est = input_nnz * (int_t)sizeof(calculate_type);
    f.index_bytes_est = input_nnz * (int_t)sizeof(idx_int);
    pangulu_Smatrix *inputs[3] = {input_0, input_1, input_2};
    for (int i = 0; i < 3; ++i) if (inputs[i])
        f.pointer_bytes_est += (inputs[i]->row + inputs[i]->column + 2) * (int_t)sizeof(int_t);
    f.output_write_bytes_est = (f.nnz_output_0 + f.nnz_output_1) * (int_t)sizeof(calculate_type);
    if (kernel == PANGULU_CALIB_SSSSM && input_0)
        f.dense_workspace_bytes_est = input_0->row * input_0->column * (int_t)sizeof(calculate_type);
    f.base_bytes_est = f.value_bytes_est + f.index_bytes_est + f.pointer_bytes_est + f.output_write_bytes_est;
    f.index_value_byte_ratio = f.value_bytes_est > 0 ? (double)(f.index_bytes_est + f.pointer_bytes_est) / (double)f.value_bytes_est : 0.0;
    f.reuse_factor_est = input_nnz > 0 ? (double)f.num_macs / (double)input_nnz : 0.0;
    /* The scheduled block is logically CSC.  CSR is an implementation-derived view. */
    pangulu_calib_index_locality(input_0 ? input_0->rowindex : NULL, input_0 ? input_0->nnz : 0,
                                 &f.block_csc_index_gap_mean, &f.block_csc_index_gap_cv,
                                 &f.block_csc_consecutive_fraction);
    pangulu_calib_index_locality(input_0 ? input_0->columnindex : NULL, input_0 ? input_0->nnz : 0,
                                 &f.derived_csr_index_gap_mean, &f.derived_csr_index_gap_cv,
                                 &f.derived_csr_consecutive_fraction);
    f.canonical_structure_hash_v2 = pangulu_calib_canonical_input_hash(kernel, input_0, input_1, input_2, output_0, output_1);
    if (kernel == PANGULU_CALIB_SSSSM && input_0 && input_0->bin_rowpointer)
    {
        int_t total = 0, maximum = 0;
        for (int i = 0; i < 10; ++i)
        {
            int_t count = input_0->bin_rowpointer[i + 2] - input_0->bin_rowpointer[i + 1];
            f.ssssm_bins[i] = count; total += count;
            if (count > 0) f.ssssm_nonempty_bins++;
            if (count > maximum) maximum = count;
        }
        f.ssssm_sparse_launches_est = f.ssssm_nonempty_bins;
        f.ssssm_max_bin_fraction = total > 0 ? (double)maximum / (double)total : 0.0;
    }
    return f;
}

static void pangulu_calib_record_task(pangulu_calib_kernel kernel,
                                      const char *variant,
                                      int_t level,
                                      int_t block_row,
                                      int_t block_col,
                                      int_t update_k,
                                      pangulu_block_common *common,
                                      const pangulu_calib_task_features *features,
                                      pangulu_calib_profile *profile)
{
    if (!pangulu_calib_is_catalog_enabled())
    {
        return;
    }

    pangulu_calib_open_catalogs(common->rank);
    FILE *catalog_fp = pangulu_calib_catalog_fps[kernel];
    if (catalog_fp == NULL)
    {
        return;
    }

    if (features == NULL) return;
    pangulu_calib_profile unknown_profile = pangulu_calib_profile_unknown(kernel);
    if (profile == NULL)
    {
        profile = &unknown_profile;
    }
    if (pangulu_calib_is_feature_replay())
    {
        profile->profile_scope = "feature_replay_pre_kernel";
    }

    double time_ms = profile->time_total_ms;
    uint64_t task_id = 1469598103934665603ULL;
    for (const char *p = pangulu_calib_matrix_id; *p != '\0'; ++p)
    {
        task_id = pangulu_calib_hash_u64(task_id, (uint64_t)(unsigned char)(*p));
    }
    task_id = pangulu_calib_hash_u64(task_id, (uint64_t)kernel);
    task_id = pangulu_calib_hash_u64(task_id, (uint64_t)level);
    task_id = pangulu_calib_hash_u64(task_id, (uint64_t)block_row);
    task_id = pangulu_calib_hash_u64(task_id, (uint64_t)block_col);
    task_id = pangulu_calib_hash_u64(task_id, (uint64_t)update_k);
    task_id = pangulu_calib_hash_u64(task_id, (uint64_t)common->rank);

    fprintf(catalog_fp,
            "%llu,%s,%s,%s,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.12g,%.12g,%.12g,%.12g,%llu,"
            "v4.1,%ld,%ld,%.12g,%ld,%ld,%ld,%.12g,%.12g,%.12g,%ld,%ld,%ld,"
            "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
            "%.12g,%.12g,%.12g,%.12g,%.12g,"
            "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
            "%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,"
            "%ld,%ld,%ld,%ld,%ld,%ld,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%ld,%ld,%llu,"
            "%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld,%.12g,%ld,%.12g,"
            "%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%d,%d,%d,%d,%ld,%ld,%ld,%d,%d\n",
            (unsigned long long)task_id,
            pangulu_calib_matrix_id,
            pangulu_calib_kernel_name(kernel),
            variant != NULL ? variant : "default",
            level,
            block_row,
            block_col,
            update_k,
            common->rank,
            common->NB,
            common->P,
            common->Q,
            features->nnz_input_0,
            features->nnz_input_1,
            features->nnz_input_2,
            features->nnz_output_0,
            features->nnz_output_1,
            features->flops,
            features->estimated_bytes,
            features->operational_intensity,
            features->row_nnz_cv,
            features->col_nnz_cv,
            features->structural_factor,
            (unsigned long long)features->structure_hash,
            features->block_rows,
            features->block_cols,
            features->aspect_ratio,
            features->min_dimension,
            features->max_dimension,
            features->main_dimension,
            features->density_input_0,
            features->density_input_1,
            features->density_input_2,
            features->num_macs,
            features->num_divisions,
            features->operation_count_v2,
            features->row_nnz_mean,
            features->row_nnz_max,
            features->row_nnz_max_mean_ratio,
            features->row_empty_fraction,
            features->col_nnz_mean,
            features->col_nnz_max,
            features->col_nnz_max_mean_ratio,
            features->col_empty_fraction,
            features->input_col_nnz_mean[0], features->input_col_nnz_max[0], features->input_col_nnz_cv[0],
            features->input_col_nnz_max_mean_ratio[0], features->input_col_empty_fraction[0],
            features->input_col_nnz_mean[1], features->input_col_nnz_max[1], features->input_col_nnz_cv[1],
            features->input_col_nnz_max_mean_ratio[1], features->input_col_empty_fraction[1],
            features->input_col_nnz_mean[2], features->input_col_nnz_max[2], features->input_col_nnz_cv[2],
            features->input_col_nnz_max_mean_ratio[2], features->input_col_empty_fraction[2],
            features->input_row_nnz_mean[0], features->input_row_nnz_max[0], features->input_row_nnz_cv[0],
            features->input_row_nnz_max_mean_ratio[0], features->input_row_empty_fraction[0],
            features->input_row_nnz_mean[1], features->input_row_nnz_max[1], features->input_row_nnz_cv[1],
            features->input_row_nnz_max_mean_ratio[1], features->input_row_empty_fraction[1],
            features->input_row_nnz_mean[2], features->input_row_nnz_max[2], features->input_row_nnz_cv[2],
            features->input_row_nnz_max_mean_ratio[2], features->input_row_empty_fraction[2],
            features->work_mean,
            features->work_max,
            features->work_cv,
            features->work_max_mean_ratio,
            features->active_work_fraction,
            features->value_bytes_est,
            features->index_bytes_est,
            features->pointer_bytes_est,
            features->output_write_bytes_est,
            features->dense_workspace_bytes_est,
            features->base_bytes_est,
            features->index_value_byte_ratio,
            features->reuse_factor_est,
            features->block_csc_index_gap_mean,
            features->block_csc_index_gap_cv,
            features->block_csc_consecutive_fraction,
            features->derived_csr_index_gap_mean,
            features->derived_csr_index_gap_cv,
            features->derived_csr_consecutive_fraction,
            features->active_rows,
            features->active_cols,
            (unsigned long long)features->canonical_structure_hash_v2,
            features->ssssm_bins[0], features->ssssm_bins[1], features->ssssm_bins[2],
            features->ssssm_bins[3], features->ssssm_bins[4], features->ssssm_bins[5],
            features->ssssm_bins[6], features->ssssm_bins[7], features->ssssm_bins[8],
            features->ssssm_bins[9], features->ssssm_nonempty_bins,
            features->ssssm_max_bin_fraction,
            features->ssssm_sparse_launches_est,
            time_ms,
            profile->profile_scope != NULL ? profile->profile_scope : "unknown",
            profile->gpu_kernel_name != NULL ? profile->gpu_kernel_name : "unknown",
            profile->time_total_ms,
            profile->time_kernel_ms,
            profile->time_memcpy_h2d_ms,
            profile->time_memcpy_d2h_ms,
            profile->time_memcpy_d2d_ms,
            profile->time_preprocess_ms,
            profile->time_overhead_ms,
            profile->num_kernel_launches,
            profile->num_memcpy_h2d,
            profile->num_memcpy_d2h,
            profile->num_memcpy_d2d,
            profile->bytes_h2d,
            profile->bytes_d2h,
            profile->bytes_d2d,
            profile->preprocess_required,
            profile->input_residency_flags);
}

#endif
