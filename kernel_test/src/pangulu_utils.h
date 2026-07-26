#ifndef PANGULU_UTILS_H
#define PANGULU_UTILS_H
#include "pangulu_common.h"

#ifdef GPU_OPEN
#include "pangulu_cuda_interface.h"
#endif

void pangulu_init_pangulu_Smatrix(pangulu_Smatrix *S)
{
    S->value = NULL;
    S->value_CSC = NULL;
    S->CSR_to_CSC_index = NULL;
    S->CSC_to_CSR_index = NULL;
    S->rowpointer = NULL;
    S->columnindex = NULL;
    S->columnpointer = NULL;
    S->rowindex = NULL;
    S->column = 0;
    S->row = 0;
    S->nnz = 0;

    S->nnzU = NULL;
    S->bin_rowpointer = NULL;
    S->bin_rowindex = NULL;

#ifdef GPU_OPEN
    S->CUDA_rowpointer = NULL;
    S->CUDA_columnindex = NULL;
    S->CUDA_value = NULL;
    S->CUDA_nnzU = NULL;
    S->CUDA_bin_rowpointer = NULL;
    S->CUDA_bin_rowindex = NULL;
#else
    S->num_lev = 0;
    S->level_idx = NULL;
    S->level_size = NULL;

#endif
}

void *pangulu_malloc(int_t size)
{

    void *malloc_address = NULL;
    malloc_address = (void *)malloc(size);
    if (malloc_address == NULL)
    {
        printf("error ------------ don't have cpu memory\n");
    }
    return malloc_address;
}
pangulu_Smatrix *pangulu_destroy_pangulu_Smatrix(pangulu_Smatrix *S)
{
    S->nnz = 0;
    S->row = 0;
    S->column = 0;
    if (S->columnpointer != NULL)
    {
        free(S->columnpointer);
    }
    if (S->rowindex != NULL)
    {
        free(S->rowindex);
    }
    if (S->value_CSC != NULL)
    {
        free(S->value_CSC);
    }
    S->columnpointer = NULL;
    S->rowindex = NULL;
    S->value_CSC = NULL;
    if (S->rowpointer != NULL)
    {
        free(S->rowpointer);
    }
    if (S->columnindex != NULL)
    {
        free(S->columnindex);
    }
    if (S->value != NULL)
    {
        free(S->value);
    }
    S->rowpointer = NULL;
    S->columnindex = NULL;
    S->value = NULL;
    if (S->bin_rowpointer != NULL)
    {
        free(S->bin_rowpointer);
    }
    S->bin_rowpointer = NULL;
    if (S->bin_rowindex != NULL)
    {
        free(S->bin_rowindex);
    }
    S->bin_rowindex = NULL;
    if (S->nnzU != NULL)
    {
        free(S->nnzU);
    }
    S->nnzU = NULL;
    if (S->CSC_to_CSR_index != NULL)
    {
        free(S->CSC_to_CSR_index);
    }
    S->CSC_to_CSR_index = NULL;
    if (S != NULL)
    {
        free(S);
    }
    return NULL;
}
int_t pangulu_bin_map(int_t nnz)
{
    int_t log_nnz = ceil(log2((double)nnz));
    if (log_nnz >= 14)
    {
        return 11;
    }
    else if (log_nnz > 5)
    {
        return log_nnz - 2;
    }
    else if (nnz > 1)
    {
        return 3;
    }
    else if (nnz == 1)
    {
        return 2;
    }
    else
    {
        return 1;
    }
}
void pangulu_Smatrix_add_more_memory_CSR(pangulu_Smatrix *A)
{
    // add CPU moemory

    int_t nnzA = A->nnz;
    int_t n = A->row;

    int_t *rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (n + 1));
    idx_int *columnindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * (nnzA));
    calculate_type *value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * (nnzA));

    int_t *CSC_to_CSR_index = (int_t *)pangulu_malloc(sizeof(int_t) * nnzA);
    for (int_t i = 0; i < nnzA; i++)
    {
        value[i] = 0.0;
    }

    for (int_t i = 0; i < (n + 1); i++)
    {
        rowpointer[i] = 0;
    }
    for (int_t i = 0; i < nnzA; i++)
    {
        rowpointer[A->rowindex[i] + 1]++;
    }
    for (int_t i = 0; i < n; i++)
    {
        rowpointer[i + 1] += rowpointer[i];
    }
    int_t *index_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (n + 1));
    for (int_t i = 0; i < n; i++)
    {
        index_rowpointer[i] = rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        for (int_t j = A->columnpointer[i]; j < A->columnpointer[i + 1]; j++)
        {

            int_t col = A->rowindex[j];
            int_t index = index_rowpointer[col];
            columnindex[index] = i;
            value[index] = A->value_CSC[j];
            CSC_to_CSR_index[j] = index;
            index_rowpointer[col]++;
        }
    }
    int_t *bin_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * BIN_LENGTH);
    int_t *bin_rowindex = (int_t *)pangulu_malloc(sizeof(int_t) * n);
    for (int_t i = 0; i < BIN_LENGTH; i++)
    {
        bin_rowpointer[i] = 0;
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t row_length = A->columnpointer[i + 1] - A->columnpointer[i];
        bin_rowpointer[pangulu_bin_map(row_length)]++;
    }
    int_t *save_bin_rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (BIN_LENGTH + 1));
    for (int_t i = 0; i < BIN_LENGTH - 1; i++)
    {
        bin_rowpointer[i + 1] += bin_rowpointer[i];
    }
    for (int_t i = 0; i < BIN_LENGTH; i++)
    {
        save_bin_rowpointer[i + 1] = bin_rowpointer[i];
    }
    for (int_t i = 0; i < n; i++)
    {
        int_t row_length = A->columnpointer[i + 1] - A->columnpointer[i];
        int_t index = pangulu_bin_map(row_length);
        bin_rowindex[save_bin_rowpointer[index]] = i;
        save_bin_rowpointer[index]++;
    }
    A->bin_rowpointer = bin_rowpointer;
    A->bin_rowindex = bin_rowindex;
    A->rowpointer = rowpointer;
    A->columnindex = columnindex;
    A->value = value;
    A->CSC_to_CSR_index = CSC_to_CSR_index;
    free(index_rowpointer);
    free(save_bin_rowpointer);

    // add GPU memory
}

#ifdef GPU_OPEN
void pangulu_destroy_cuda_memory_pangulu_Smatrix(pangulu_Smatrix *S)
{
    if (S->CUDA_rowpointer != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_rowpointer);
    }
    S->CUDA_rowpointer = NULL;
    if (S->CUDA_columnindex != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_columnindex);
    }
    S->CUDA_columnindex = NULL;
    if (S->CUDA_value != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_value);
    }
    S->CUDA_value = NULL;
    if (S->CUDA_bin_rowpointer != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_bin_rowpointer);
    }
    S->CUDA_bin_rowpointer = NULL;
    if (S->CUDA_bin_rowindex != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_bin_rowindex);
    }
    S->CUDA_bin_rowindex = NULL;
    if (S->CUDA_nnzU != NULL)
    {
        pangulu_cuda_free_interface(S->CUDA_nnzU);
    }
    S->CUDA_nnzU = NULL;
}
#endif

int_t pangulu_binary_read_csr_pangulu_Smatrix(pangulu_Smatrix *S, char *filename)
{
    int_t nrow, ncol;
    int_t nnz;
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("binary file %s don't exit\n", filename);
        return -1;
    }
    if (1 != fread(&nrow, sizeof(int_t), 1, fp))
    {
        printf("read nrow error\n");
        return -1;
    }

    if (1 != fread(&ncol, sizeof(int_t), 1, fp))
    {
        printf("read ncol error\n");
        return -1;
    }
    if (1 != fread(&nnz, sizeof(int_t), 1, fp))
    {
        printf("read nnz error\n");
        return -1;
    }
    int_t *rowpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (nrow + 1));
    if ((unsigned int_t)(nrow + 1) != fread(rowpointer, sizeof(int_t), nrow + 1, fp))
    {
        printf("read rowptr error\n");
        return -1;
    }
    // printf("nnz = %ld\n",nnz);
    idx_int *columnindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * nnz);
    calculate_type *value = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * nnz);

    if ((unsigned int_t)nnz != fread(columnindex, sizeof(idx_int), nnz, fp))
    {
        printf("read columnindex error\n");
        return -1;
    }
    if ((unsigned int_t)nnz != fread(value, sizeof(calculate_type), nnz, fp))
    {
        printf("read value error\n");
        return -1;
    }

    S->row = nrow;
    S->column = ncol;
    S->rowpointer = rowpointer;
    S->columnindex = columnindex;
    S->value = value;
    S->nnz = nnz;
    fclose(fp);

    return 0;
}
int_t pangulu_binary_read_csc_pangulu_Smatrix(pangulu_Smatrix *S, char *filename)
{
    int_t nrow, ncol;
    int_t nnz;
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("binary file %s don't exit\n", filename);
        return -1;
    }
    if (1 != fread(&nrow, sizeof(int_t), 1, fp))
    {
        printf("read nrow error\n");
        return -1;
    }

    if (1 != fread(&ncol, sizeof(int_t), 1, fp))
    {
        printf("read ncol error\n");
        return -1;
    }
    if (1 != fread(&nnz, sizeof(int_t), 1, fp))
    {
        printf("read nnz error\n");
        return -1;
    }
    int_t *columnpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (nrow + 1));
    if ((unsigned int_t)(nrow + 1) != fread(columnpointer, sizeof(int_t), nrow + 1, fp))
    {
        printf("read rowptr error\n");
        return -1;
    }
    // printf("nnz = %ld\n",nnz);
    idx_int *rowindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * nnz);
    calculate_type *value_CSC = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * nnz);

    if ((unsigned int_t)nnz != fread(rowindex, sizeof(idx_int), nnz, fp))
    {
        printf("read rowindex error\n");
        return -1;
    }
    // if((unsigned int_t)nnz!=fread(value_CSC,sizeof(calculate_type),nnz,fp)){
    //     printf("read value_CSC error\n");
    //     return -1;
    // }
    for (int i = 0; i < nnz; i++)
    {
        value_CSC[i] = 1.0;
    }

    S->row = nrow;
    S->column = ncol;
    S->columnpointer = columnpointer;
    S->rowindex = rowindex;
    S->value_CSC = value_CSC;
    S->nnz = nnz;

    fclose(fp);
    pangulu_Smatrix_add_more_memory_CSR(S); // add memory

#ifdef GPU_OPEN
    pangulu_Smatrix_add_CUDA_memory(S);
    pangulu_Smatrix_CUDA_memcpy_A(S);
#endif

    return 0;
}
int_t pangulu_binary_read_csc_value_pangulu_Smatrix(pangulu_Smatrix *S, char *filename)
{
    int_t nrow, ncol;
    int_t nnz;
    FILE *fp = NULL;
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        printf("binary file %s don't exit\n", filename);
        return -1;
    }
    if (1 != fread(&nrow, sizeof(int_t), 1, fp))
    {
        printf("read nrow error\n");
        return -1;
    }

    if (1 != fread(&ncol, sizeof(int_t), 1, fp))
    {
        printf("read ncol error\n");
        return -1;
    }
    if (1 != fread(&nnz, sizeof(int_t), 1, fp))
    {
        printf("read nnz error\n");
        return -1;
    }
    int_t *columnpointer = (int_t *)pangulu_malloc(sizeof(int_t) * (nrow + 1));
    if ((unsigned int_t)(nrow + 1) != fread(columnpointer, sizeof(int_t), nrow + 1, fp))
    {
        printf("read rowptr error\n");
        return -1;
    }
    // printf("nnz = %ld\n",nnz);
    idx_int *rowindex = (idx_int *)pangulu_malloc(sizeof(idx_int) * nnz);
    calculate_type *value_CSC = (calculate_type *)pangulu_malloc(sizeof(calculate_type) * nnz);

    if ((unsigned int_t)nnz != fread(rowindex, sizeof(idx_int), nnz, fp))
    {
        printf("read rowindex error\n");
        return -1;
    }
    if ((unsigned int_t)nnz != fread(value_CSC, sizeof(calculate_type), nnz, fp))
    {
        printf("read value_CSC error\n");
        return -1;
    }
    // for(int i = 0; i < nnz; i++){
    //     value_CSC[i] = 1.0;
    // }

    S->row = nrow;
    S->column = ncol;
    S->columnpointer = columnpointer;
    S->rowindex = rowindex;
    S->value_CSC = value_CSC;
    S->nnz = nnz;

    fclose(fp);
    pangulu_Smatrix_add_more_memory_CSR(S); // add memory
#ifdef GPU_OPEN
    pangulu_Smatrix_add_CUDA_memory(S);
    pangulu_Smatrix_CUDA_memcpy_A(S);

#endif
    return 0;
}

void pangulu_Smatrix_add_memory_U(pangulu_Smatrix *U)
{
    int_32t *nnzU = (int_32t *)pangulu_malloc(sizeof(int_32t) * (U->row));
    for (int_t j = 0; j < U->row; j++)
    {
        nnzU[j] = U->columnpointer[j + 1] - U->columnpointer[j] - 1;
    }
    U->nnzU = nnzU;
}

int_t Binary_search_position(int_t begin, int_t end, int_t aim, idx_int *array)
{
    int_t middle = (end + begin) / 2;
    int_t left = begin;
    int_t right = end;
    while (left <= right)
    {
        if (array[middle] > aim)
        {
            right = middle - 1;
        }
        else if (array[middle] < aim)
        {
            left = middle + 1;
        }
        else
        {
            return middle;
        }
        middle = (right + left) / 2;
    }
    return -1; // not find
}

void pangulu_display_pangulu_Smatrix_CSC(pangulu_Smatrix *S)
{
    printf("------------------\n\n\n");
    if (S == NULL)
    {
        printf("\nno i am null\n");
        return;
    }
    printf("row is %ld column is %ld\n", S->row, S->column);
    printf("columnpointer:");
    for (int_t i = 0; i < S->row + 1; i++)
    {
        printf("%ld ", S->columnpointer[i]);
    }
    printf("\n");
    printf("rowindex:\n");
    for (int_t i = 0; i < S->row; i++)
    {
        for (int_t j = S->columnpointer[i]; j < S->columnpointer[i + 1]; j++)
        {
            printf("%d ", S->rowindex[j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("value_csc:\n");
    for (int_t i = 0; i < S->row; i++)
    {
        // printf("i %ld\n",i);
        for (int_t j = S->columnpointer[i]; j < S->columnpointer[i + 1]; j++)
        {
            printf("%lf ", S->value_CSC[j]);
        }
        printf("\n");
    }
    printf("\n\n\n--------------------");
}

void pangulu_display_pangulu_Smatrix_CSR(pangulu_Smatrix *S)
{
    printf("------------------\n\n\n");
    if (S == NULL)
    {
        printf("\nno i am null\n");
        return;
    }
    printf("row is %ld column is %ld\n", S->row, S->column);
    printf("rowpointer:");
    for (int_t i = 0; i < S->row + 1; i++)
    {
        printf("%ld ", S->rowpointer[i]);
    }
    printf("\n");
    printf("columnindex:\n");
    for (int_t i = 0; i < S->row; i++)
    {
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            printf("%d ", S->columnindex[j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("value_csr:\n");
    for (int_t i = 0; i < S->row; i++)
    {
        // printf("i %ld\n",i);
        for (int_t j = S->rowpointer[i]; j < S->rowpointer[i + 1]; j++)
        {
            printf("%lf ", S->value[j]);
        }
        printf("\n");
    }
    printf("\n\n\n--------------------");
}

void pangulu_pangulu_Smatrix_memcpy_value_CSC_copy_length(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    for (int_t i = 0; i < copy_S->nnz; i++)
    {
        S->value_CSC[i] = copy_S->value_CSC[i];
    }
}
void pangulu_pangulu_Smatrix_memcpy_value_CSR_copy_length(pangulu_Smatrix *S, pangulu_Smatrix *copy_S)
{
    for (int_t i = 0; i < copy_S->nnz; i++)
    {
        S->value[i] = copy_S->value[i];
    }
}

#ifdef GPU_OPEN
void TRIANGLE_PRE_CPU(idx_int *L_rowindex,
                      const int_t n,
                      const int_t nnzL,
                      int *d_graphInDegree)
{
    for (int i = 0; i < nnzL; i++)
    {
        d_graphInDegree[L_rowindex[i]] += 1;
    }
}

void pangulu_gessm_preprocess(pangulu_Smatrix *L)
{
    int_t n = L->row;
    int_t nnzL = L->nnz;

    /**********************************L****************************************/

    int *graphInDegree = L->graphInDegree;
    memset(graphInDegree, 0, n * sizeof(int));

    TRIANGLE_PRE_CPU(L->rowindex, n, nnzL, graphInDegree);
}

void pangulu_tstrf_preprocess(pangulu_Smatrix *U)
{
    int_t n = U->row;
    int_t nnzU = U->nnz;

    /**********************************L****************************************/

    int *graphInDegree = U->graphInDegree;
    memset(graphInDegree, 0, n * sizeof(int));

    TRIANGLE_PRE_CPU(U->columnindex, n, nnzU, graphInDegree);
}
#endif
void pangulu_transport_pangulu_Smatrix_CSC_to_CSR(pangulu_Smatrix *S)
{
    for (int_t i = 0; i < S->nnz; i++)
    {
        int_t index = S->CSC_to_CSR_index[i];
        S->value[index] = S->value_CSC[i];
    }
}
void pangulu_transport_pangulu_Smatrix_CSR_to_CSC(pangulu_Smatrix *S)
{
    for (int_t i = 0; i < S->nnz; i++)
    {
        int_t index = S->CSC_to_CSR_index[i];
        S->value_CSC[i] = S->value[index];
    }
}

void pangulu_init_value(pangulu_Smatrix *S, calculate_type init_value)
{
    for (int i = 0; i < S->nnz; i++)
    {
        S->value[i] = init_value;
        S->value_CSC[i] = init_value;
    }
}

void pangulu_time_check_begin()
{
    gettimeofday(&(GET_TIME_START), NULL);
}

double pangulu_time_check_end()
{
    gettimeofday(&(GET_TIME_END), NULL);
    return ((GET_TIME_END.tv_sec - GET_TIME_START.tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START.tv_usec) / 1000.0);
}
pangulu_kernel_time *pangulu_kernel_time_init()
{
    pangulu_kernel_time *time = (pangulu_kernel_time *)malloc(sizeof(pangulu_kernel_time));
    time->SSSSM_CPU_DENSE_TIME = 0;
    time->SSSSM_CPU_THMKL_TIME = 0;
    time->SSSSM_CPU_SPMM_TIME = 0;
    time->SSSSM_GPU_DENSE_TIME = 0;
    time->SSSSM_GPU_SPARSE_TIME = 0;
    time->GETRF_CPU_DENSE_TIME = 0;
    time->GETRF_CPU_SPARSE_TIME = 0;
    time->GETRF_GPU_DENSE_TIME = 0;
    time->GETRF_GPU_SPARSE_TIME = 0;
    time->GESSM_CPU_1_TIME = 0;
    time->GESSM_CPU_2_TIME = 0;
    time->GESSM_CPU_3_TIME = 0;
    time->GESSM_CPU_4_TIME = 0;
    time->GESSM_CPU_5_TIME = 0;
    time->GESSM_CPU_6_TIME = 0;
    time->GESSM_GPU_7_TIME = 0;
    time->GESSM_GPU_8_TIME = 0;
    time->GESSM_GPU_9_TIME = 0;
    time->GESSM_GPU_10_TIME = 0;
    time->GESSM_GPU_11_TIME = 0;
    time->TSTRF_CPU_1_TIME = 0;
    time->TSTRF_CPU_2_TIME = 0;
    time->TSTRF_CPU_3_TIME = 0;
    time->TSTRF_CPU_4_TIME = 0;
    time->TSTRF_CPU_5_TIME = 0;
    time->TSTRF_CPU_6_TIME = 0;
    time->TSTRF_GPU_7_TIME = 0;
    time->TSTRF_GPU_8_TIME = 0;
    time->TSTRF_GPU_9_TIME = 0;
    time->TSTRF_GPU_10_TIME = 0;
    time->TSTRF_GPU_11_TIME = 0;
    return time;
}
void pangulu_kernel_time_free(pangulu_kernel_time *time)
{
    if (time != NULL)
    {
        free(time);
    }
}
void pangulu_CSC_copy(pangulu_Smatrix *src, pangulu_Smatrix *dest)
{
    for (int i = 0; i < src->nnz; i++)
    {
        dest->value_CSC[i] = src->value_CSC[i];
    }
}
void pangulu_CSR_copy(pangulu_Smatrix *src, pangulu_Smatrix *dest)
{
    for (int i = 0; i < src->nnz; i++)
    {
        dest->value[i] = src->value[i];
    }
}
// int max_row_nnz;
// int min_row_nnz_C;
// int max_col_nnz_C;
// int min_col_nnz_C;
// double average_row_nnz_C;
// double average_col_nnz_C;
// double density;
int max_data(int a, int b)
{
    return a > b ? a : b;
}
int min_data(int a, int b)
{
    return a < b ? a : b;
}
#ifdef KERNEL_PRINT_INFO
void calculate_matrix_info(pangulu_Smatrix *A, matrix_info *info)
{
    int n = A->row;
    int nnz = A->nnz;
    double density = (double)nnz / (double)(n * n);
    int max_row_nnz_A = -1;                             // 每一行的最多非零元
    int min_row_nnz_A = 0x7fffffff;                     // 每一行的最少非零元
    int max_col_nnz_A = -1;                             // 每一列的最多非零元
    int min_col_nnz_A = 0x7fffffff;                     // 每一列的最少非零元
    double average_row_nnz_A = (double)nnz / (double)n; // 平均每行元素数量
    double average_col_nnz_A = (double)nnz / (double)n; // 平均每列元素数量
    for (int i = 0; i < n; i++)
    {
        int row_nnz_num = A->rowpointer[i + 1] - A->rowpointer[i];
        int col_nnz_num = A->columnpointer[i + 1] - A->columnpointer[i];
        max_row_nnz_A = max_data(row_nnz_num, max_row_nnz_A);
        min_row_nnz_A = min_data(row_nnz_num, min_row_nnz_A);
        max_col_nnz_A = max_data(col_nnz_num, max_col_nnz_A);
        min_col_nnz_A = min_data(col_nnz_num, min_col_nnz_A);
    }
    double variety_row = 0;
    double variety_col = 0;
    for (int i = 0; i < n; i++)
    {
        int row_nnz_num = A->rowpointer[i + 1] - A->rowpointer[i];
        int col_nnz_num = A->columnpointer[i + 1] - A->columnpointer[i];
        variety_row += (average_row_nnz_A - row_nnz_num) * (average_row_nnz_A - row_nnz_num);
        variety_col += (average_col_nnz_A - col_nnz_num) * (average_col_nnz_A - col_nnz_num);
    }
    variety_row /= average_row_nnz_A;
    variety_col /= average_col_nnz_A;
    info->n = n;
    info->nnz = nnz;
    info->max_col_nnz = max_col_nnz_A;
    info->min_col_nnz = min_col_nnz_A;
    info->max_row_nnz = max_row_nnz_A;
    info->min_row_nnz = min_row_nnz_A;
    info->average_row_nnz = average_row_nnz_A;
    info->average_col_nnz = average_col_nnz_A;
    info->density = density;
    info->variety_col = variety_col;
    info->variety_row = variety_row;
}
void printInfo(pangulu_Smatrix *A, pangulu_kernel_time *time, Pangulu_print_kernel_type type)
{
    int n = A->row;
    int nnz = A->nnz;
    double density = (double)nnz / (double)(n * n);
    int max_row_nnz = -1;                               // 每一行的最多非零元
    int min_row_nnz_C = 0x7fffffff;                     // 每一行的最少非零元
    int max_col_nnz_C = -1;                             // 每一列的最多非零元
    int min_col_nnz_C = 0x7fffffff;                     // 每一列的最少非零元
    double average_row_nnz_C = (double)nnz / (double)n; // 平均每行元素数量
    double average_col_nnz_C = (double)nnz / (double)n; // 平均每列元素数量
    for (int i = 0; i < n; i++)
    {
        int row_nnz_num = A->rowpointer[i + 1] - A->rowpointer[i];
        int col_nnz_num = A->columnpointer[i + 1] - A->columnpointer[i];
        max_row_nnz = max_data(row_nnz_num, max_row_nnz);
        min_row_nnz_C = min_data(row_nnz_num, min_row_nnz_C);
        max_col_nnz_C = max_data(col_nnz_num, max_col_nnz_C);
        min_col_nnz_C = min_data(col_nnz_num, min_col_nnz_C);
    }
    double variety_row = 0;
    double variety_col = 0;
    for (int i = 0; i < n; i++)
    {
        int row_nnz_num = A->rowpointer[i + 1] - A->rowpointer[i];
        int col_nnz_num = A->columnpointer[i + 1] - A->columnpointer[i];
        variety_row += (average_row_nnz_C - row_nnz_num) * (average_row_nnz_C - row_nnz_num);
        variety_col += (average_col_nnz_C - col_nnz_num) * (average_col_nnz_C - col_nnz_num);
    }
    variety_row /= average_row_nnz_C;
    variety_col /= average_col_nnz_C;
    char file_name[512];
    if (type == SSSSM)
    {
        sprintf(file_name, "%s/%s.csv", OUTPUT_FILE_PATH, "SSSSM");
        FILE *fp = fopen(file_name, "a");
        fprintf(fp, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
                n, nnz, max_row_nnz, min_row_nnz_C, max_col_nnz_C, min_col_nnz_C,
                average_row_nnz_C, average_col_nnz_C, variety_row, variety_col,
                time->SSSSM_CPU_DENSE_TIME, time->SSSSM_CPU_THMKL_TIME, time->SSSSM_GPU_DENSE_TIME, time->SSSSM_GPU_SPARSE_TIME);
        fclose(fp);
    }
    else if (type == GETRF)
    {
        sprintf(file_name, "%s/%s.csv", OUTPUT_FILE_PATH, "GETRF");
        FILE *fp = fopen(file_name, "a");
        fprintf(fp, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
                n, nnz, max_row_nnz, min_row_nnz_C, max_col_nnz_C, min_col_nnz_C,
                average_row_nnz_C, average_col_nnz_C, variety_row, variety_col,
                time->GETRF_CPU_DENSE_TIME, time->GETRF_GPU_DENSE_TIME, time->GETRF_GPU_SPARSE_TIME);
        fclose(fp);
    }
    else if (type == GESSM)
    {
        sprintf(file_name, "%s/%s.csv", OUTPUT_FILE_PATH, "GESSM");
        FILE *fp = fopen(file_name, "a");
        fprintf(fp, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
                n, nnz, max_row_nnz, min_row_nnz_C, max_col_nnz_C, min_col_nnz_C,
                average_row_nnz_C, average_col_nnz_C, variety_row, variety_col,
                time->GESSM_CPU_1_TIME, time->GESSM_CPU_2_TIME, time->GESSM_CPU_3_TIME, time->GESSM_CPU_4_TIME, time->GESSM_CPU_5_TIME, time->GESSM_CPU_6_TIME, time->GESSM_GPU_7_TIME, time->GESSM_GPU_8_TIME, time->GESSM_GPU_9_TIME, time->GESSM_GPU_10_TIME, time->GESSM_GPU_11_TIME);
        fclose(fp);
    }
    else if (type == TSTRF)
    {
        sprintf(file_name, "%s/%s.csv", OUTPUT_FILE_PATH, "TSTRF");
        FILE *fp = fopen(file_name, "a");
        fprintf(fp, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
                n, nnz, max_row_nnz, min_row_nnz_C, max_col_nnz_C, min_col_nnz_C,
                average_row_nnz_C, average_col_nnz_C, variety_row, variety_col,
                time->TSTRF_CPU_1_TIME, time->TSTRF_CPU_2_TIME, time->TSTRF_CPU_3_TIME, time->TSTRF_CPU_4_TIME, time->TSTRF_CPU_5_TIME, time->TSTRF_CPU_6_TIME, time->TSTRF_GPU_7_TIME, time->TSTRF_GPU_8_TIME, time->TSTRF_GPU_9_TIME, time->TSTRF_GPU_10_TIME, time->TSTRF_GPU_11_TIME);
        fclose(fp);
    }
}
void printInfo_GETRF(pangulu_Smatrix *A, pangulu_kernel_time *time)
{
    int n = A->row;
    int nnz = A->nnz;
    double density = (double)nnz / (double)(n * n);
    int max_row_nnz = -1;                               // 每一行的最多非零元
    int min_row_nnz_C = 0x7fffffff;                     // 每一行的最少非零元
    int max_col_nnz_C = -1;                             // 每一列的最多非零元
    int min_col_nnz_C = 0x7fffffff;                     // 每一列的最少非零元
    double average_row_nnz_C = (double)nnz / (double)n; // 平均每行元素数量
    double average_col_nnz_C = (double)nnz / (double)n; // 平均每列元素数量
    for (int i = 0; i < n; i++)
    {
        int row_nnz_num = A->rowpointer[i + 1] - A->rowpointer[i];
        int col_nnz_num = A->columnpointer[i + 1] - A->columnpointer[i];
        max_row_nnz = max_data(row_nnz_num, max_row_nnz);
        min_row_nnz_C = min_data(row_nnz_num, min_row_nnz_C);
        max_col_nnz_C = max_data(col_nnz_num, max_col_nnz_C);
        min_col_nnz_C = min_data(col_nnz_num, min_col_nnz_C);
    }
    double variety_row = 0;
    double variety_col = 0;
    for (int i = 0; i < n; i++)
    {
        int row_nnz_num = A->rowpointer[i + 1] - A->rowpointer[i];
        int col_nnz_num = A->columnpointer[i + 1] - A->columnpointer[i];
        variety_row += (average_row_nnz_C - row_nnz_num) * (average_row_nnz_C - row_nnz_num);
        variety_col += (average_col_nnz_C - col_nnz_num) * (average_col_nnz_C - col_nnz_num);
    }
    variety_row /= average_row_nnz_C;
    variety_col /= average_col_nnz_C;
    char file_name[512];
    sprintf(file_name, "%s/%s.csv", OUTPUT_FILE_PATH, "figure7_GETRF");
    FILE *fp = fopen(file_name, "a");
    fprintf(fp, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
            n, nnz, max_row_nnz, min_row_nnz_C, max_col_nnz_C, min_col_nnz_C,
            average_row_nnz_C, average_col_nnz_C, variety_row, variety_col,
            time->GETRF_CPU_DENSE_TIME, time->GETRF_GPU_DENSE_TIME, time->GETRF_GPU_SPARSE_TIME);
    fclose(fp);
}

void printInfo_SSSSM(pangulu_Smatrix *A, pangulu_Smatrix *B, pangulu_Smatrix *C, pangulu_kernel_time *time)
{
    int n = C->row;
    int nnz_C = C->nnz;
    int nnz_A = A->nnz;
    int nnz_B = B->nnz;
    double density = (double)nnz_C / (double)(n * n);
    int max_row_nnz_C = -1;                               // 每一行的最多非零元
    int min_row_nnz_C = 0x7fffffff;                       // 每一行的最少非零元
    int max_col_nnz_C = -1;                               // 每一列的最多非零元
    int min_col_nnz_C = 0x7fffffff;                       // 每一列的最少非零元
    double average_row_nnz_C = (double)nnz_C / (double)n; // 平均每行元素数量
    double average_col_nnz_C = (double)nnz_C / (double)n; // 平均每列元素数量
    long long int calcalution_sum = 0;                    // 总运算量
    for (int i = 0; i < n; i++)
    {
        int row_nnz_num = C->rowpointer[i + 1] - C->rowpointer[i];
        int col_nnz_num = C->columnpointer[i + 1] - C->columnpointer[i];
        max_row_nnz_C = max_data(row_nnz_num, max_row_nnz_C);
        min_row_nnz_C = min_data(row_nnz_num, min_row_nnz_C);
        max_col_nnz_C = max_data(col_nnz_num, max_col_nnz_C);
        min_col_nnz_C = min_data(col_nnz_num, min_col_nnz_C);
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = B->columnpointer[i]; j < B->columnpointer[i + 1]; j++)
        {
            int col_a = B->rowindex[j];
            calcalution_sum += A->columnpointer[col_a + 1] - A->columnpointer[col_a];
        }
    }
    double variety_row = 0;
    double variety_col = 0;
    for (int i = 0; i < n; i++)
    {
        int row_nnz_num = C->rowpointer[i + 1] - C->rowpointer[i];
        int col_nnz_num = C->columnpointer[i + 1] - C->columnpointer[i];
        variety_row += (average_row_nnz_C - row_nnz_num) * (average_row_nnz_C - row_nnz_num);
        variety_col += (average_col_nnz_C - col_nnz_num) * (average_col_nnz_C - col_nnz_num);
    }
    variety_row /= average_row_nnz_C;
    variety_col /= average_col_nnz_C;
    char file_name[512];
    sprintf(file_name, "%s/%s.csv", OUTPUT_FILE_PATH, "figure7_SSSSM");
    FILE *fp = fopen(file_name, "a");
    fprintf(fp, "%d,%d,%d,%d,%lld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
            n, nnz_A, nnz_B, nnz_C, calcalution_sum, max_row_nnz_C, min_row_nnz_C, max_col_nnz_C, min_col_nnz_C,
            average_row_nnz_C, average_col_nnz_C, variety_row, variety_col,
            time->SSSSM_CPU_DENSE_TIME, time->SSSSM_CPU_THMKL_TIME, time->SSSSM_GPU_DENSE_TIME, time->SSSSM_GPU_SPARSE_TIME);
    fclose(fp);
}
void printInfo_GESSM(pangulu_Smatrix *L, pangulu_Smatrix *B, pangulu_kernel_time *time)
{
    matrix_info *L_info = (matrix_info *)malloc(sizeof(matrix_info));
    matrix_info *B_info = (matrix_info *)malloc(sizeof(matrix_info));
    calculate_matrix_info(L, L_info);
    calculate_matrix_info(B, B_info);
    char file_name[512];
    sprintf(file_name, "%s/%s.csv", OUTPUT_FILE_PATH, "figure7_GESSM");
    FILE *fp = fopen(file_name, "a");
    fprintf(fp, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
            B_info->n, B_info->nnz, B_info->max_row_nnz, B_info->min_row_nnz, B_info->max_col_nnz, B_info->min_col_nnz,
            B_info->average_row_nnz, B_info->average_col_nnz, B_info->variety_row, B_info->variety_col, L_info->nnz, L_info->max_row_nnz, L_info->min_row_nnz, L_info->max_col_nnz, L_info->min_col_nnz,
            L_info->average_row_nnz, L_info->average_col_nnz, L_info->variety_row, L_info->variety_col,
            time->GESSM_CPU_1_TIME, time->GESSM_CPU_2_TIME, time->GESSM_CPU_3_TIME, time->GESSM_CPU_4_TIME, time->GESSM_CPU_5_TIME, time->GESSM_CPU_6_TIME, time->GESSM_GPU_7_TIME, time->GESSM_GPU_8_TIME, time->GESSM_GPU_9_TIME, time->GESSM_GPU_10_TIME, time->GESSM_GPU_11_TIME);
    free(L_info);
    free(B_info);
    fclose(fp);
}

void printInfo_TSTRF(pangulu_Smatrix *U, pangulu_Smatrix *B, pangulu_kernel_time *time)
{
    matrix_info *U_info = (matrix_info *)malloc(sizeof(matrix_info));
    matrix_info *B_info = (matrix_info *)malloc(sizeof(matrix_info));
    calculate_matrix_info(U, U_info);
    calculate_matrix_info(B, B_info);
    char file_name[512];
    sprintf(file_name, "%s/%s.csv", OUTPUT_FILE_PATH, "figure7_TSTRF");
    FILE *fp = fopen(file_name, "a");
    fprintf(fp, "%d,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
            B_info->n, B_info->nnz, B_info->max_row_nnz, B_info->min_row_nnz, B_info->max_col_nnz, B_info->min_col_nnz,
            B_info->average_row_nnz, B_info->average_col_nnz, B_info->variety_row, B_info->variety_col, U_info->nnz, U_info->max_row_nnz, U_info->min_row_nnz, U_info->max_col_nnz, U_info->min_col_nnz,
            U_info->average_row_nnz, U_info->average_col_nnz, U_info->variety_row, U_info->variety_col,
            time->TSTRF_CPU_1_TIME, time->TSTRF_CPU_2_TIME, time->TSTRF_CPU_3_TIME, time->TSTRF_CPU_4_TIME, time->TSTRF_CPU_5_TIME, time->TSTRF_CPU_6_TIME, time->TSTRF_GPU_7_TIME, time->TSTRF_GPU_8_TIME, time->TSTRF_GPU_9_TIME, time->TSTRF_GPU_10_TIME, time->TSTRF_GPU_11_TIME);
    free(U_info);
    free(B_info);
    fclose(fp);
}
#endif
int get_test_count(double time)
{
    if (time > 100.0)
    {
        return 1;
    }
    else if (time > 50.0)
    {
        return 2;
    }
    else if (time > 30.0)
    {
        return 4;
    }
    else if (time > 10.0)
    {
        return 8;
    }
    else if (time > 5.0)
    {
        return 20;
    }
    else if (time > 1.0)
    {
        return 50;
    }
    else
    {
        return 100;
    }
}

#endif