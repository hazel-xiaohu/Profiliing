#include "pangulu_tstrf_fp64.h"
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
    double tstrf_time;
    char *B_filename = NULL;
    char *U_filename = NULL;
    char *X_filename = NULL;
    pangulu_Smatrix *B = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *U = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *X = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *TEMP_B = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix *TEMP_X = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_init_pangulu_Smatrix(B);
    pangulu_init_pangulu_Smatrix(U);
    pangulu_init_pangulu_Smatrix(X);
    pangulu_init_pangulu_Smatrix(TEMP_B);
    pangulu_init_pangulu_Smatrix(TEMP_X);

    int arg_index = 0;
    while (arg_index < argc)
    {
        if (strcmp(argv[arg_index], "-B") == 0)
        {
            arg_index++;
            B_filename = argv[arg_index++];
            pangulu_binary_read_csc_pangulu_Smatrix(B, B_filename);
            pangulu_binary_read_csc_pangulu_Smatrix(X, B_filename);
            pangulu_binary_read_csc_pangulu_Smatrix(TEMP_B, B_filename);
            pangulu_binary_read_csc_pangulu_Smatrix(TEMP_X, B_filename);
#ifdef GPU_OPEN
            pangulu_cuda_malloc((void **)&(B->d_left_sum), ((B->nnz)) * sizeof(calculate_type));
#endif
        }
        else if (strcmp(argv[arg_index], "-U") == 0)
        {
            arg_index++;
            U_filename = argv[arg_index++];
            pangulu_binary_read_csc_pangulu_Smatrix(U, U_filename);
#ifdef GPU_OPEN
            pangulu_cuda_malloc((void **)&(U->d_left_sum), ((U->nnz)) * sizeof(calculate_type));
            pangulu_cuda_malloc((void **)&(U->d_graphInDegree), U->row * sizeof(int));
            pangulu_cuda_malloc((void **)&(U->d_id_extractor), sizeof(int));
            U->graphInDegree = (int *)pangulu_malloc(U->row * sizeof(int));
            pangulu_tstrf_preprocess(U);
#endif
        }
        else
        {
            arg_index++;
        }
    }
    int n = B->row;

    TEMP_A_value = (calculate_type *)malloc(sizeof(calculate_type) * n * n);
#ifdef GPU_OPEN
    pangulu_cuda_malloc((void **)&CUDA_TEMP_value, n * n * sizeof(calculate_type));
#endif
    double time_cpu_addition = 0;
    double time_gpu_addition = 0;
    pangulu_time_check_begin();
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(B, B);
    time_cpu_addition = pangulu_time_check_end();

    pangulu_time_check_begin();
    pangulu_Smatrix_CUDA_memcpy_value_CSC(B, B);
    time_gpu_addition = pangulu_time_check_end();
    int test_count = 0;
    double test_time = 0;
    int temp_count = 0;

#ifdef TSTRF_CPU_1_TEST // 最终结果在B中
    pangulu_CSC_copy(TEMP_B, B);
    pangulu_CSC_copy(TEMP_B, X);
    pangulu_time_check_begin();
    pangulu_tstrf_interface(B, X, U, TSTRF_CPU_1);
    time->TSTRF_CPU_1_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_CPU_1_TIME);
    test_time = time->TSTRF_CPU_1_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_CSC_copy(TEMP_B, B);
        pangulu_CSC_copy(TEMP_B, X);
        pangulu_time_check_begin();
        pangulu_tstrf_interface(B, X, U, TSTRF_CPU_1);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_CPU_1_TIME = (test_time / (double)test_count) + time_cpu_addition;

#ifdef CHECK_KERNEL_ERROR
    check_tstrf_error_interface(B, X, U, TSTRF_CPU_1);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_CPU_1_TIME:\t%lf ms\n\n", time->TSTRF_CPU_1_TIME);
#endif
#endif

#ifdef TSTRF_CPU_2_TEST // 最终结果在B中
    pangulu_CSC_copy(TEMP_B, B);
    pangulu_CSC_copy(TEMP_B, X);
    pangulu_time_check_begin();
    pangulu_tstrf_interface(B, X, U, TSTRF_CPU_2);
    time->TSTRF_CPU_2_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_CPU_2_TIME);
    test_time = time->TSTRF_CPU_2_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_CSC_copy(TEMP_B, B);
        pangulu_CSC_copy(TEMP_B, X);
        pangulu_time_check_begin();
        pangulu_tstrf_interface(B, X, U, TSTRF_CPU_2);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_CPU_2_TIME = (test_time / (double)test_count) + time_cpu_addition;
#ifdef CHECK_KERNEL_ERROR
    check_tstrf_error_interface(B, X, U, TSTRF_CPU_2);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_CPU_2_TIME:\t%lf ms\n\n", time->TSTRF_CPU_2_TIME);
#endif
#endif

#ifdef TSTRF_CPU_3_TEST // 最终结果在B中
    pangulu_CSR_copy(TEMP_B, B);
    pangulu_CSR_copy(TEMP_B, X);
    pangulu_time_check_begin();
    pangulu_tstrf_interface(B, X, U, TSTRF_CPU_3);
    time->TSTRF_CPU_3_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_CPU_3_TIME);
    test_time = time->TSTRF_CPU_3_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_CSC_copy(TEMP_B, B);
        pangulu_CSC_copy(TEMP_B, X);
        pangulu_time_check_begin();
        pangulu_tstrf_interface(B, X, U, TSTRF_CPU_3);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_CPU_3_TIME = (test_time / (double)test_count) + time_cpu_addition;
#ifdef CHECK_KERNEL_ERROR
    check_tstrf_error_interface(B, X, U, TSTRF_CPU_3);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_CPU_3_TIME:\t%lf ms\n\n", time->TSTRF_CPU_3_TIME);
#endif
#endif

#ifdef TSTRF_CPU_4_TEST // 最终结果在B中
    pangulu_CSR_copy(TEMP_B, B);
    pangulu_CSR_copy(TEMP_B, X);
    pangulu_time_check_begin();
    pangulu_tstrf_interface(B, X, U, TSTRF_CPU_4);
    time->TSTRF_CPU_4_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_CPU_4_TIME);
    test_time = time->TSTRF_CPU_4_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_CSC_copy(TEMP_B, B);
        pangulu_CSC_copy(TEMP_B, X);
        pangulu_time_check_begin();
        pangulu_tstrf_interface(B, X, U, TSTRF_CPU_4);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_CPU_4_TIME = (test_time / (double)test_count) + time_cpu_addition;
#ifdef CHECK_KERNEL_ERROR
    check_tstrf_error_interface(B, X, U, TSTRF_CPU_4);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_CPU_4_TIME:\t%lf ms\n\n", time->TSTRF_CPU_4_TIME);
#endif
#endif

#ifdef TSTRF_CPU_5_TEST // 最终结果在B中
    pangulu_CSR_copy(TEMP_B, B);
    pangulu_CSR_copy(TEMP_B, X);
    pangulu_time_check_begin();
    pangulu_tstrf_interface(B, X, U, TSTRF_CPU_5);
    time->TSTRF_CPU_5_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_CPU_5_TIME);
    test_time = time->TSTRF_CPU_5_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_CSC_copy(TEMP_B, B);
        pangulu_CSC_copy(TEMP_B, X);
        pangulu_time_check_begin();
        pangulu_tstrf_interface(B, X, U, TSTRF_CPU_5);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_CPU_5_TIME = (test_time / (double)test_count) + time_cpu_addition;
#ifdef CHECK_KERNEL_ERROR
    check_tstrf_error_interface(B, X, U, TSTRF_CPU_5);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_CPU_5_TIME:\t%lf ms\n\n", time->TSTRF_CPU_5_TIME);
#endif
#endif

#ifdef TSTRF_CPU_6_TEST // 最终结果在B中
    pangulu_CSR_copy(TEMP_B, B);
    pangulu_CSR_copy(TEMP_B, X);
    pangulu_time_check_begin();
    pangulu_tstrf_interface(B, X, U, TSTRF_CPU_6);
    time->TSTRF_CPU_6_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_CPU_6_TIME);
    test_time = time->TSTRF_CPU_6_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_CSC_copy(TEMP_B, B);
        pangulu_CSC_copy(TEMP_B, X);
        pangulu_time_check_begin();
        pangulu_tstrf_interface(B, X, U, TSTRF_CPU_6);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_CPU_6_TIME = (test_time / (double)test_count) + time_cpu_addition;
#ifdef CHECK_KERNEL_ERROR
    check_tstrf_error_interface(B, X, U, TSTRF_CPU_6);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_CPU_6_TIME:\t%lf ms\n\n", time->TSTRF_CPU_6_TIME);
#endif
#endif

#ifdef GPU_OPEN

#ifdef TSTRF_GPU_7_TEST // 都是CSR

    pangulu_Smatrix_CUDA_memcpy_complete_CSR(X, TEMP_X);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
    pangulu_time_check_begin();
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(B, TEMP_B);
    pangulu_tstrf_interface(B, X, U, TSTRF_GPU_7);
    time->TSTRF_GPU_7_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_GPU_7_TIME);
    test_time = time->TSTRF_GPU_7_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(X, TEMP_X);
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
        pangulu_time_check_begin();
        pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
        pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(B, TEMP_B);
        pangulu_tstrf_interface(B, X, U, TSTRF_GPU_7);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_GPU_7_TIME = (test_time / (double)test_count) + time_gpu_addition;

#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSR(B, X);
    check_tstrf_error_interface(B, X, U, TSTRF_GPU_7);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_GPU_7_TIME:\t%lf ms\n\n", time->TSTRF_GPU_7_TIME);
#endif
#endif

#ifdef TSTRF_GPU_8_TEST // X是CSC，B是CSC，U是CSR
    pangulu_Smatrix_CUDA_memcpy_complete_CSC(B, TEMP_B);
    pangulu_Smatrix_CUDA_memcpy_complete_CSC(X, TEMP_X);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
    pangulu_time_check_begin();
    pangulu_tstrf_interface(B, X, U, TSTRF_GPU_8);
    time->TSTRF_GPU_8_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_GPU_8_TIME);
    test_time = time->TSTRF_GPU_8_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_Smatrix_CUDA_memcpy_complete_CSC(B, TEMP_B);
        pangulu_Smatrix_CUDA_memcpy_complete_CSC(X, TEMP_X);
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
        pangulu_time_check_begin();
        pangulu_tstrf_interface(B, X, U, TSTRF_GPU_8);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_GPU_8_TIME = (test_time / (double)test_count) + time_gpu_addition;
#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSC(B, X);
    check_tstrf_error_interface(B, X, U, TSTRF_GPU_8);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_GPU_8_TIME:\t%lf ms\n\n", time->TSTRF_GPU_8_TIME);
#endif
#endif

#ifdef TSTRF_GPU_9_TEST // 都是CSR
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(X, TEMP_X);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
    pangulu_time_check_begin();
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(B, TEMP_B);
    pangulu_tstrf_interface(B, X, U, TSTRF_GPU_9);
    time->TSTRF_GPU_9_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_GPU_9_TIME);
    test_time = time->TSTRF_GPU_9_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {

        pangulu_Smatrix_CUDA_memcpy_complete_CSR(X, TEMP_X);
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
        pangulu_time_check_begin();
        pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
        pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(B, TEMP_B);
        pangulu_tstrf_interface(B, X, U, TSTRF_GPU_9);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_GPU_9_TIME = (test_time / (double)test_count) + time_gpu_addition;
#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSR(B, X);
    check_tstrf_error_interface(B, X, U, TSTRF_GPU_9);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_GPU_9_TIME:\t%lf ms\n\n", time->TSTRF_GPU_9_TIME);
#endif
#endif

#ifdef TSTRF_GPU_10_TEST // 都是CSR
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(X, TEMP_X);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
    pangulu_time_check_begin();
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
    pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(B, TEMP_B);
    pangulu_tstrf_interface(B, X, U, TSTRF_GPU_10);
    time->TSTRF_GPU_10_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_GPU_10_TIME);
    test_time = time->TSTRF_GPU_10_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(X, TEMP_X);
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
        pangulu_time_check_begin();
        pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
        pangulu_transport_pangulu_Smatrix_CSC_to_CSR(B);
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(B, TEMP_B);
        pangulu_tstrf_interface(B, X, U, TSTRF_GPU_10);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_GPU_10_TIME = (test_time / (double)test_count) + time_gpu_addition;
#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSR(B, X);
    check_tstrf_error_interface(B, X, U, TSTRF_GPU_10);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_GPU_10_TIME:\t%lf ms\n\n", time->TSTRF_GPU_10_TIME);
#endif
#endif

#ifdef TSTRF_GPU_11_TEST // B是CSC X是CSC U是CSR
    pangulu_Smatrix_CUDA_memcpy_complete_CSC(B, TEMP_B);
    pangulu_Smatrix_CUDA_memcpy_complete_CSC(X, TEMP_X);
    pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
    pangulu_time_check_begin();
    pangulu_tstrf_interface(B, X, U, TSTRF_GPU_11);
    time->TSTRF_GPU_11_TIME = pangulu_time_check_end();

    test_count = get_test_count(time->TSTRF_GPU_11_TIME);
    test_time = time->TSTRF_GPU_11_TIME;
    temp_count = test_count;

    while ((temp_count--) > 1)
    {
        pangulu_Smatrix_CUDA_memcpy_complete_CSC(B, TEMP_B);
        pangulu_Smatrix_CUDA_memcpy_complete_CSC(X, TEMP_X);
        pangulu_Smatrix_CUDA_memcpy_complete_CSR(U, U);
        pangulu_time_check_begin();
        pangulu_tstrf_interface(B, X, U, TSTRF_GPU_11);
        test_time += pangulu_time_check_end();
    }
    time->TSTRF_GPU_11_TIME = (test_time / (double)test_count) + time_gpu_addition;

#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSC(B, X);
    check_tstrf_error_interface(B, X, U, TSTRF_GPU_11);
#endif
#ifdef PRINT_TIME
    printf("TSTRF_GPU_11_TIME:\t%lf ms\n\n", time->TSTRF_GPU_11_TIME);
#endif
#endif
#endif

#ifdef KERNEL_PRINT_INFO
    printInfo_TSTRF(U, B, time);
#endif

    pangulu_destroy_pangulu_Smatrix(B);
    pangulu_destroy_pangulu_Smatrix(U);
    pangulu_destroy_pangulu_Smatrix(X);
    pangulu_destroy_pangulu_Smatrix(TEMP_X);
    pangulu_destroy_pangulu_Smatrix(TEMP_B);
#ifdef GPU_OPEN
    pangulu_cuda_free_interface(CUDA_TEMP_value);
#endif
    free(TEMP_A_value);
    return 0;
}