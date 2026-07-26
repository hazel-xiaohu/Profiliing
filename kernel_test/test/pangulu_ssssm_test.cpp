#include "pangulu_ssssm_fp64.h"
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
int main(int argc,char **argv){
    double ssssm_time;
    char *A_filename = NULL;
    char *B_filename = NULL;
    char *C_filename = NULL;
    pangulu_Smatrix * A = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix * TEMP_C = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix * B = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_Smatrix * C = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_init_pangulu_Smatrix(A);
    pangulu_init_pangulu_Smatrix(TEMP_C);
    pangulu_init_pangulu_Smatrix(B);
    pangulu_init_pangulu_Smatrix(C);

    int arg_index = 0;
    while(arg_index < argc){
        if (strcmp(argv[arg_index], "-A") == 0){
            arg_index ++;
            A_filename = argv[arg_index++];
            pangulu_binary_read_csc_pangulu_Smatrix(A,A_filename);
        }else if(strcmp(argv[arg_index], "-B") == 0){
            arg_index ++;
            B_filename = argv[arg_index++];    
            pangulu_binary_read_csc_pangulu_Smatrix(B,B_filename);
        }else if(strcmp(argv[arg_index], "-C") == 0){
            arg_index ++;
            C_filename = argv[arg_index++];    
            pangulu_binary_read_csc_pangulu_Smatrix(C,C_filename);
            pangulu_binary_read_csc_pangulu_Smatrix(TEMP_C,C_filename);
        }
        else{
            arg_index++;
        }       
    } 
    int n = A -> row;

    TEMP_A_value = (calculate_type*)malloc(sizeof(calculate_type)*n*n);
#ifdef GPU_OPEN
    pangulu_cuda_malloc((void**)&CUDA_TEMP_value, n * n * sizeof(calculate_type));
    pangulu_cuda_malloc((void**)&CUDA_B_idx_COL, n * n * sizeof(idx_int));
#endif

    double time_cup_addition = 0;
    double time_gpu_addition = 0;
    pangulu_time_check_begin();
    // pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(A, A);
    time_cup_addition = pangulu_time_check_end();

    pangulu_time_check_begin();
    pangulu_Smatrix_CUDA_memcpy_value_CSC(A, A);
    time_gpu_addition = pangulu_time_check_end();

    int test_count = 0;
    double test_time = 0;
    int temp_count = 0;

#ifdef SSSSM_CPU_DENSE_TEST
    pangulu_CSC_copy(TEMP_C,C);
    pangulu_time_check_begin();  
    pangulu_ssssm_interface(C,A,B,SSSSM_CPU_DENSE);
    time -> SSSSM_CPU_DENSE_TIME  = pangulu_time_check_end(); 

    test_count = get_test_count(time -> SSSSM_CPU_DENSE_TIME);
    test_time = time -> SSSSM_CPU_DENSE_TIME;
    temp_count = test_count;

    while((temp_count--) > 1){
        pangulu_CSC_copy(TEMP_C,C);
        pangulu_time_check_begin();  
        pangulu_ssssm_interface(C,A,B,SSSSM_CPU_DENSE);
        test_time += pangulu_time_check_end(); 
    }
    time->SSSSM_CPU_DENSE_TIME = (test_time / (double)test_count) + time_cup_addition;

#ifdef CHECK_KERNEL_ERROR
    check_ans_ssssm(C,A,B);
#endif
#ifdef PRINT_TIME
    printf("SSSSM_CPU_DENSE_TIME:\t%lf ms\n\n",time -> SSSSM_CPU_DENSE_TIME);
#endif
#endif

#ifdef SSSSM_CPU_THMKL_TEST
    pangulu_CSC_copy(TEMP_C,C);
    pangulu_time_check_begin();  
    pangulu_ssssm_interface(C,A,B,SSSSM_CPU_THMKL);
    time -> SSSSM_CPU_THMKL_TIME  = pangulu_time_check_end();  

    test_count = get_test_count(time -> SSSSM_CPU_THMKL_TIME);
    test_time  = time -> SSSSM_CPU_THMKL_TIME;
    temp_count = test_count;

    while((temp_count--) > 1){
        pangulu_CSC_copy(TEMP_C,C);
        pangulu_time_check_begin();  
        pangulu_ssssm_interface(C,A,B,SSSSM_CPU_DENSE);
        test_time += pangulu_time_check_end(); 
    }
    time->SSSSM_CPU_THMKL_TIME = (test_time / (double)test_count) + time_cup_addition;

#ifdef CHECK_KERNEL_ERROR
    check_ans_ssssm(C,A,B);
#endif
#ifdef PRINT_TIME
    printf("SSSSM_CPU_THMKL_TIME:\t%lf ms\n\n",time -> SSSSM_CPU_THMKL_TIME);
#endif
#endif

// #ifdef SSSSM_CPU_SPMM_TEST
//     pangulu_CSC_copy(TEMP_C,C);
//     pangulu_time_check_begin();  
//     pangulu_ssssm_interface(C,A,B,SSSSM_CPU_SPMM);
//     time -> SSSSM_CPU_SPMM_TIME  = pangulu_time_check_end();  
// #ifdef CHECK_KERNEL_ERROR
//     check_ans_ssssm(C,A,B);
// #endif
// #ifdef PRINT_TIME
//     printf("SSSSM_CPU_SPMM_TIME:\t%lf ms\n\n",time -> SSSSM_CPU_SPMM_TIME);
// #endif
// #endif

#ifdef GPU_OPEN

#ifdef SSSSM_GPU_DENSE_TEST
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(C,TEMP_C);
    pangulu_time_check_begin();  
    pangulu_ssssm_interface(C,A,B,SSSSM_GPU_DENSE);
    time -> SSSSM_GPU_DENSE_TIME  = pangulu_time_check_end();  

    test_count = get_test_count(time -> SSSSM_GPU_DENSE_TIME);
    test_time = time -> SSSSM_GPU_DENSE_TIME;
    temp_count = test_count;

    while((temp_count--) > 1){
        pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(C,TEMP_C);
        pangulu_time_check_begin();  
        pangulu_ssssm_interface(C,A,B,SSSSM_CPU_DENSE);
        test_time += pangulu_time_check_end(); 
    }
    time->SSSSM_GPU_DENSE_TIME = (test_time / (double)test_count) + time_gpu_addition;

#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSC(C,C);
    check_ans_ssssm(C,A,B);
#endif
#ifdef PRINT_TIME 
    printf("SSSSM_GPU_DENSE_TIME:\t%lf ms\n\n",time -> SSSSM_GPU_DENSE_TIME);
#endif
#endif

#ifdef SSSSM_GPU_SPARSE_TEST
    pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(C,TEMP_C);
    pangulu_time_check_begin();  
    pangulu_ssssm_interface(C,A,B,SSSSM_GPU_SPARSE);
    time -> SSSSM_GPU_SPARSE_TIME  = pangulu_time_check_end();  

    test_count = get_test_count(time -> SSSSM_GPU_SPARSE_TIME);
    test_time = time -> SSSSM_GPU_SPARSE_TIME;
    temp_count = test_count;

    while((temp_count--) > 1){
        pangulu_Smatrix_CUDA_memcpy_to_device_value_CSC(C,TEMP_C);
        pangulu_time_check_begin();  
        pangulu_ssssm_interface(C,A,B,SSSSM_CPU_DENSE);
        test_time += pangulu_time_check_end(); 
    }
    time->SSSSM_GPU_SPARSE_TIME = (test_time / (double)test_count) + time_gpu_addition;

#ifdef CHECK_KERNEL_ERROR
    pangulu_Smatrix_CUDA_memcpy_value_CSC(C,C);
    check_ans_ssssm(C,A,B);
#endif

#ifdef PRINT_TIME
    printf("SSSSM_GPU_SPARSE_TIME:\t%lf ms\n\n",time -> SSSSM_GPU_SPARSE_TIME);
#endif
#endif

#endif

#ifdef KERNEL_PRINT_INFO
    printInfo_SSSSM(A,B,C,time);
#endif

    
    pangulu_destroy_pangulu_Smatrix(A);
    pangulu_destroy_pangulu_Smatrix(TEMP_C);
    pangulu_destroy_pangulu_Smatrix(B);
    pangulu_destroy_pangulu_Smatrix(C);

    free(TEMP_A_value);

    free(time);

#ifdef GPU_OPEN
    pangulu_cuda_free_interface(CUDA_B_idx_COL);
    pangulu_cuda_free_interface(CUDA_TEMP_value);
#endif
    return 0;
}