#include "pangulu_getrf_fp64.h"
#include "pangulu_utils.h"
#include "pangulu_check_answer.h"
#ifdef GPU_OPEN
#include "pangulu_cuda.h"
#endif
calculate_type *TEMP_A_value = NULL;
struct timeval GET_TIME_START, GET_TIME_END;
#ifdef GPU_OPEN
idx_int *CUDA_B_idx_COL = NULL;
calculate_type *CUDA_TEMP_value = NULL;
#endif
pangulu_kernel_time *time = pangulu_kernel_time_init();
int main(int argc, char **argv)
{
    double getrf_time;
    char *A_filename = NULL;
    char *L_filename = NULL;
    char *U_filename = NULL;
    pangulu_Smatrix *A = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *L = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *U = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *TEMP_A = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *TEMP_L = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *TEMP_U = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));

    pangulu_init_pangulu_Smatrix(A);
    pangulu_init_pangulu_Smatrix(L);
    pangulu_init_pangulu_Smatrix(U);
    pangulu_init_pangulu_Smatrix(TEMP_A);
    pangulu_init_pangulu_Smatrix(TEMP_L);
    pangulu_init_pangulu_Smatrix(TEMP_U);

    int arg_index = 0;
    while (arg_index < argc)
    {
        if (strcmp(argv[arg_index], "-A") == 0)
        {
            arg_index++;
            A_filename = argv[arg_index++];
            pangulu_binary_read_csc_value_pangulu_Smatrix(A, A_filename);
            pangulu_binary_read_csc_value_pangulu_Smatrix(TEMP_A, A_filename);
        }
        else if (strcmp(argv[arg_index], "-L") == 0)
        {
            arg_index++;
            L_filename = argv[arg_index++];
            pangulu_binary_read_csc_pangulu_Smatrix(L, L_filename);
            pangulu_binary_read_csc_pangulu_Smatrix(TEMP_L, L_filename);
        }
        else if (strcmp(argv[arg_index], "-U") == 0)
        {
            arg_index++;
            U_filename = argv[arg_index++];
            pangulu_binary_read_csc_pangulu_Smatrix(U, U_filename);
            pangulu_binary_read_csc_pangulu_Smatrix(TEMP_U, U_filename);
#ifdef GPU_OPEN
            pangulu_Smatrix_add_memory_U(U);
            pangulu_Smatrix_add_CUDA_memory_U(U);
            pangulu_Smatrix_CUDA_memcpy_nnzU(U, U);
            pangulu_Smatrix_add_memory_U(TEMP_U);
            pangulu_Smatrix_add_CUDA_memory_U(TEMP_U);
            pangulu_Smatrix_CUDA_memcpy_nnzU(TEMP_U, TEMP_U);
#endif
        }
        else
        {
            arg_index++;
        }
    }

    int n = A->row;

    TEMP_A_value = (calculate_type *)malloc(sizeof(calculate_type) * n * n);

#ifdef GPU_OPEN
    pangulu_cuda_malloc((void **)&CUDA_TEMP_value, n * n * sizeof(calculate_type));
#endif

    double time_cpu_addition = 0;
    double time_gpu_addition = 0;
    // pangulu_time_check_begin();
    // // pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, A);
    // time_cpu_addition = pangulu_time_check_end();

    pangulu_time_check_begin();
    // pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
    time_gpu_addition = pangulu_time_check_end();

    int test_count = 0;
    double test_time = 0;
    int temp_count = 0;

    // #ifdef GETRF_CPU_SPARSE_TEST
    //     pangulu_time_check_begin();
    //     pangulu_getrf_interface(A,L,U,GETRF_CPU_SPARSE);
    //     time -> GETRF_CPU_SPARSE_TIME  = pangulu_time_check_end();
    //     pangulu_CSC_copy(TEMP_A,A);
    //     check_ans_getrf(A,L,U);
    // #ifdef PRINT_TIME
    //     printf("GETRF_CPU_SPARSE_TIME:\t%lf ms\n\n",time -> GETRF_CPU_SPARSE_TIME);
    // #endif
    // #endif

#ifdef GETRF_CPU_DENSE_TEST
    pangulu_CSC_copy(TEMP_A, A);
    pangulu_time_check_begin();
    pangulu_getrf_interface(A, L, U, GETRF_CPU_DENSE);
    time->GETRF_CPU_DENSE_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->GETRF_CPU_DENSE_TIME);
    test_time = time->GETRF_CPU_DENSE_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_CSC_copy(TEMP_A, A);
        pangulu_time_check_begin();
        pangulu_getrf_interface(A, L, U, GETRF_CPU_DENSE);
        test_time += pangulu_time_check_end();
    }
    time->GETRF_CPU_DENSE_TIME = (test_time / (double)test_count) + time_cpu_addition;

#ifdef CHECK_KERNEL_ERROR
    check_ans_getrf(A, L, U);
#endif

#ifdef PRINT_TIME
    printf("GETRF_CPU_DENSE_TIME:\t%lf ms\n\n", time->GETRF_CPU_DENSE_TIME);
#endif
#endif

#ifdef GPU_OPEN

#ifdef GETRF_GPU_DENSE_TEST
    pangulu_Smatrix_CUDA_memcpy_nnzU(U, TEMP_U);
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, TEMP_A);
    pangulu_time_check_begin();
    pangulu_getrf_interface(A, L, U, GETRF_GPU_DENSE);
    time->GETRF_GPU_DENSE_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->GETRF_GPU_DENSE_TIME);
    test_time = time->GETRF_GPU_DENSE_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_Smatrix_CUDA_memcpy_nnzU(U, TEMP_U);
        pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, TEMP_A);
        pangulu_time_check_begin();
        pangulu_getrf_interface(A, L, U, GETRF_GPU_DENSE);
        test_time += pangulu_time_check_end();
    }
    time->GETRF_GPU_DENSE_TIME = (test_time / (double)test_count) + time_gpu_addition;

#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, U);
    check_ans_getrf(A, L, U);
#endif
#ifdef PRINT_TIME
    printf("GETRF_GPU_DENSE_TIME:\t%lf ms\n\n", time->GETRF_GPU_DENSE_TIME);
#endif
#endif

#ifdef GETRF_GPU_SPARSE_TEST
    pangulu_Smatrix_CUDA_memcpy_nnzU(U, TEMP_U);
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, TEMP_A);
    pangulu_time_check_begin();
    pangulu_getrf_interface(A, L, U, GETRF_GPU_SPARSE);
    time->GETRF_GPU_SPARSE_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->GETRF_GPU_SPARSE_TIME);
    test_time = time->GETRF_GPU_SPARSE_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_Smatrix_CUDA_memcpy_nnzU(U, TEMP_U);
        pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, TEMP_A);
        pangulu_time_check_begin();
        pangulu_getrf_interface(A, L, U, GETRF_GPU_SPARSE);
        test_time += pangulu_time_check_end();
    }
    time->GETRF_GPU_SPARSE_TIME = (test_time / (double)test_count) + time_gpu_addition;

#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, U);
    check_ans_getrf(A, L, U);
#endif
#ifdef PRINT_TIME
    printf("GETRF_GPU_SPARSE_TIME:\t%lf ms\n\n", time->GETRF_GPU_SPARSE_TIME);
#endif

#endif

#endif

#ifdef KERNEL_PRINT_INFO
    printInfo_GETRF(A, time);
#endif

    pangulu_destroy_pangulu_Smatrix(A);
    pangulu_destroy_pangulu_Smatrix(L);
    pangulu_destroy_pangulu_Smatrix(U);

#ifdef GPU_OPEN
    pangulu_cuda_free_interface(CUDA_TEMP_value);
#endif
    free(TEMP_A_value);
    return 0;
}