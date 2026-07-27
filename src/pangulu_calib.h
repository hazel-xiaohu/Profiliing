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

static FILE *pangulu_calib_catalog_fp = NULL;
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

static int pangulu_calib_is_catalog_enabled(void)
{
    if (pangulu_calib_catalog_enabled >= 0)
    {
        return pangulu_calib_catalog_enabled;
    }

    const char *mode = getenv("CALIB_MODE");
    pangulu_calib_catalog_enabled = (mode != NULL && strcmp(mode, "catalog") == 0) ? 1 : 0;
    return pangulu_calib_catalog_enabled;
}

static void pangulu_calib_open_catalog(int_32t rank)
{
    if (!pangulu_calib_is_catalog_enabled() || pangulu_calib_catalog_fp != NULL)
    {
        return;
    }

    const char *dir = getenv("CALIB_CATALOG_DIR");
    if (dir == NULL || dir[0] == '\0')
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

    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.rank%d.catalog.csv", dir, pangulu_calib_matrix_id, rank);
    pangulu_calib_catalog_fp = fopen(path, "w");
    if (pangulu_calib_catalog_fp == NULL)
    {
        fprintf(stderr, "PanguLU catalog: failed to open %s\n", path);
        pangulu_calib_catalog_enabled = 0;
        return;
    }

    fprintf(pangulu_calib_catalog_fp,
            "task_id,matrix_id,kernel_type,kernel_variant,level,block_row,block_col,update_k,rank,block_size,"
            "process_rows,process_cols,"
            "nnz_input_0,nnz_input_1,nnz_output,flops,estimated_bytes,operational_intensity,"
            "row_nnz_cv,col_nnz_cv,structural_factor,structure_hash,time_ms,"
            "profile_scope,gpu_kernel_name,time_total_ms,time_kernel_ms,time_memcpy_h2d_ms,"
            "time_memcpy_d2h_ms,time_memcpy_d2d_ms,time_preprocess_ms,time_overhead_ms,"
            "num_kernel_launches,num_memcpy_h2d,num_memcpy_d2h,num_memcpy_d2d,"
            "bytes_h2d,bytes_d2h,bytes_d2d,preprocess_required,input_residency_flags\n");
}

static void pangulu_calib_close_catalog(void)
{
    if (pangulu_calib_catalog_fp != NULL)
    {
        fclose(pangulu_calib_catalog_fp);
        pangulu_calib_catalog_fp = NULL;
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

static uint64_t pangulu_calib_structure_hash(pangulu_Smatrix *A, pangulu_Smatrix *B, pangulu_Smatrix *C)
{
    uint64_t hash = 1469598103934665603ULL;
    pangulu_Smatrix *matrices[3] = {A, B, C};
    for (int i = 0; i < 3; ++i)
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

static void pangulu_calib_record_task(pangulu_calib_kernel kernel,
                                      const char *variant,
                                      int_t level,
                                      int_t block_row,
                                      int_t block_col,
                                      int_t update_k,
                                      pangulu_block_common *common,
                                      pangulu_Smatrix *A,
                                      pangulu_Smatrix *B,
                                      pangulu_Smatrix *C,
                                      pangulu_calib_profile *profile)
{
    if (!pangulu_calib_is_catalog_enabled())
    {
        return;
    }

    pangulu_calib_open_catalog(common->rank);
    if (pangulu_calib_catalog_fp == NULL)
    {
        return;
    }

    int_t macs = 0;
    double row_cv = 0.0;
    double col_cv = 0.0;
    double structural_factor = 0.0;
    if (kernel == PANGULU_CALIB_GETRF)
    {
        macs = pangulu_calib_macs_lu(B, C);
        row_cv = pangulu_calib_cv_from_ptr(A != NULL ? A->rowpointer : NULL, A != NULL ? A->row : 0);
        col_cv = pangulu_calib_cv_from_ptr(A != NULL ? A->columnpointer : NULL, A != NULL ? A->column : 0);
        structural_factor = PANGULU_MAX(row_cv, col_cv);
    }
    else if (kernel == PANGULU_CALIB_TSTRF)
    {
        macs = pangulu_calib_macs_axu(A, B);
        row_cv = pangulu_calib_cv_from_ptr(B != NULL ? B->rowpointer : NULL, B != NULL ? B->row : 0);
        col_cv = pangulu_calib_cv_from_ptr(A != NULL ? A->columnpointer : NULL, A != NULL ? A->column : 0);
        structural_factor = row_cv;
    }
    else if (kernel == PANGULU_CALIB_GESSM)
    {
        macs = pangulu_calib_macs_lxa(A, B);
        row_cv = pangulu_calib_cv_from_ptr(A != NULL ? A->rowpointer : NULL, A != NULL ? A->row : 0);
        col_cv = pangulu_calib_cv_from_ptr(B != NULL ? B->columnpointer : NULL, B != NULL ? B->column : 0);
        structural_factor = col_cv;
    }
    else
    {
        macs = pangulu_calib_macs_lu(B, C);
        row_cv = pangulu_calib_cv_from_ptr(C != NULL ? C->rowpointer : NULL, C != NULL ? C->row : 0);
        col_cv = pangulu_calib_cv_from_ptr(B != NULL ? B->columnpointer : NULL, B != NULL ? B->column : 0);
        structural_factor = (A != NULL && A->nnz > 0) ? ((double)macs / (double)A->nnz) : 0.0;
    }

    int_t nnz0 = A != NULL ? A->nnz : 0;
    int_t nnz1 = B != NULL ? B->nnz : 0;
    int_t nnz2 = C != NULL ? C->nnz : 0;
    int_t flops = macs * 2;
    int_t estimated_bytes = (nnz0 + nnz1 + nnz2) * (int_t)(sizeof(calculate_type) + sizeof(idx_int));
    double oi = estimated_bytes > 0 ? (double)flops / (double)estimated_bytes : 0.0;

    uint64_t structure_hash = pangulu_calib_structure_hash(A, B, C);
    pangulu_calib_profile unknown_profile = pangulu_calib_profile_unknown(kernel);
    if (profile == NULL)
    {
        profile = &unknown_profile;
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

    fprintf(pangulu_calib_catalog_fp,
            "%llu,%s,%s,%s,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%ld,%ld,%ld,%ld,%ld,%.12g,%.12g,%.12g,%.12g,%llu,%.12g,"
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
            nnz0,
            nnz1,
            nnz2,
            flops,
            estimated_bytes,
            oi,
            row_cv,
            col_cv,
            structural_factor,
            (unsigned long long)structure_hash,
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
