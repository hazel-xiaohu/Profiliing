#ifndef PANGULU_CHECK_ANSWER_H
#define PANGULU_CHECK_ANSWER_H

#include "pangulu_common.h"
#include "pangulu_utils.h"
#define INIT_VALUE 1.0

void check_ans_ssssm(pangulu_Smatrix *C,pangulu_Smatrix *A,pangulu_Smatrix *B)
{
    double error = 0;
    for(int i = 0;i < B->row; i++){
        for(int j = B->columnpointer[i];j < B->columnpointer[i+1];j++){
            int k = B->rowindex[j];
            calculate_type value_B = INIT_VALUE;
            for(int l = A->columnpointer[k];l < A->columnpointer[k+1]; l++){
                int row = A->rowindex[l];
                int C_col_begin = C->columnpointer[i];
                int C_col_end = C->columnpointer[i+1];
                int pos = Binary_search_position(0,  C_col_end - C_col_begin - 1, row,(C->rowindex) + C_col_begin); 
                if(pos != -1){
                    C->value_CSC[C_col_begin + pos] += INIT_VALUE;
                }   
            }
        }
    }
    for(int i = 0;i < C->nnz;i++){
        error += (double)fabs(C->value_CSC[i] - INIT_VALUE);
    }
    if(error!=0){
        printf("ssssm error:%le\n",error);
    }
}

void check_ans_getrf(pangulu_Smatrix *A,pangulu_Smatrix *L,pangulu_Smatrix *U)
{
    double error = 0;
    calculate_type *temp_value = (calculate_type*)malloc(sizeof(calculate_type) * A->nnz);
    memcpy(temp_value,A->value_CSC,sizeof(calculate_type) * A->nnz);
    for(int i = 0;i < U->row; i++){
        for(int j = U->columnpointer[i];j < U->columnpointer[i+1];j++){
            int k = U->rowindex[j];
            calculate_type value_U = U->value_CSC[j];
            for(int l = L->columnpointer[k];l < L->columnpointer[k+1]; l++){
                int row = L->rowindex[l];
                int A_col_begin = A->columnpointer[i];
                int A_col_end = A->columnpointer[i+1];
                int pos = Binary_search_position(0,  A_col_end - A_col_begin - 1, row,(A->rowindex) + A_col_begin); 
                if(pos != -1){
                    temp_value[A_col_begin + pos] -= value_U * L->value_CSC[l];
                }   
            }
        }
    }
    for(int i = 0;i < A->nnz;i++){
        error += (double)fabs(temp_value[i] - 0.0);
    }
    if(error > 1e-14){
        printf("getrf error:%le\n",error);
    }
    free(temp_value);
}

void check_gessm_error(pangulu_Smatrix *B,pangulu_Smatrix *L,pangulu_Smatrix *X)
{
    double error = 0;
    calculate_type *temp_value = (calculate_type*)malloc(sizeof(calculate_type) * B->nnz);
    memcpy(temp_value,B->value_CSC,sizeof(calculate_type) * B->nnz);
    for(int i = 0;i < X->row; i++){
        for(int j = X->columnpointer[i];j < X->columnpointer[i+1];j++){
            int k = X->rowindex[j];
            calculate_type value_X = X->value_CSC[j];
            for(int l = L->columnpointer[k];l < L->columnpointer[k+1]; l++){
                int row = L->rowindex[l];
                int B_col_begin = B->columnpointer[i];
                int B_col_end = B->columnpointer[i+1];
                int pos = Binary_search_position(0,  B_col_end - B_col_begin - 1, row,(B->rowindex) + B_col_begin); 
                if(pos != -1){
                    temp_value[B_col_begin + pos] -= value_X * L->value_CSC[l];
                }   
            }
        }
    }
    for(int i = 0;i < B->nnz;i++){
        error += (double)fabs(temp_value[i] - 0.0);
    }
    if(error>0){
    printf("gessm error:%le\n",error);
    }

    free(temp_value);
}

void check_tstrf_error(pangulu_Smatrix *B,pangulu_Smatrix *X,pangulu_Smatrix *U)
{
    double error = 0;
    calculate_type *temp_value = (calculate_type*)malloc(sizeof(calculate_type) * B->nnz);
    memcpy(temp_value,B->value_CSC,sizeof(calculate_type) * B->nnz);
    for(int i = 0;i < U->row; i++){
        for(int j = U->columnpointer[i];j < U->columnpointer[i+1];j++){
            int k = U->rowindex[j];
            calculate_type value_U = U->value_CSC[j];
            for(int l = X->columnpointer[k];l < X->columnpointer[k+1]; l++){
                int row = X->rowindex[l];
                int B_col_begin = B->columnpointer[i];
                int B_col_end = B->columnpointer[i+1];
                int pos = Binary_search_position(0,  B_col_end - B_col_begin - 1, row,(B->rowindex) + B_col_begin); 
                if(pos != -1){
                    temp_value[B_col_begin + pos] -= value_U * X->value_CSC[l];
                }   
            }
        }
    }
    for(int i = 0;i < B->nnz;i++){
        error += (double)fabs(temp_value[i] - 0.0);
    }
    if(error>0){
    printf("tstrf error:%le\n",error);
    }
    free(temp_value);
}

void check_gessm_error_interface(pangulu_Smatrix *B,pangulu_Smatrix *L,pangulu_Smatrix *X,Pangulu_kernel_type kernel_type){
    pangulu_init_value(X,INIT_VALUE);
    switch(kernel_type){
        case GESSM_CPU_1:
        case GESSM_CPU_5:
        case GESSM_GPU_8:
        case GESSM_GPU_11:
             pangulu_transport_pangulu_Smatrix_CSR_to_CSC(B);
             check_gessm_error(X, L, B);
             break;
        case GESSM_CPU_2:
        case GESSM_CPU_3:
        case GESSM_CPU_4:
        case GESSM_CPU_6:
        case GESSM_GPU_7:
        case GESSM_GPU_9:
        case GESSM_GPU_10:
             check_gessm_error(X, L, B);
             break;
        default:
            printf("Invalid kernel name!\n");
    }

}
void check_tstrf_error_interface(pangulu_Smatrix *B,pangulu_Smatrix *X,pangulu_Smatrix *U,Pangulu_kernel_type kernel_type){
    pangulu_init_value(X,INIT_VALUE);
    switch(kernel_type){
        case TSTRF_CPU_1:
        case TSTRF_CPU_5:
        case TSTRF_GPU_8:
        case TSTRF_GPU_11:
             check_tstrf_error(X, B, U);
             break;
        case TSTRF_CPU_2:
        case TSTRF_CPU_3:
        case TSTRF_CPU_4:
        case TSTRF_CPU_6:
        case TSTRF_GPU_7:
        case TSTRF_GPU_9:
        case TSTRF_GPU_10:
             pangulu_transport_pangulu_Smatrix_CSR_to_CSC(B);
             check_tstrf_error(X, B, U);
             break;
        default:
            printf("Invalid kernel name!\n");
    }

}
#endif