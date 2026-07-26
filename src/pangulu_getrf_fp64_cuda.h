#ifndef PANGULU_GETRF_FP64_CUDA_H
#define PANGULU_GETRF_FP64_CUDA_H

#include "pangulu_common.h"
#include "pangulu_cuda.h"
#include "pangulu_cuda_interface.h"


void pangulu_getrf_fp64_cuda(pangulu_Smatrix *A,
                             pangulu_Smatrix *L,
                             pangulu_Smatrix *U)
{

    if (A->row == A->nnz)
    {
        calculate_type *tmp = (calculate_type *)malloc(sizeof(calculate_type) * A->nnz);
        for (int i = 0; i < A->row; i++)
        {
            tmp[i] = 1.0;
        }
        cudaMemcpy(L->CUDA_value, tmp, A->nnz * sizeof(calculate_type), cudaMemcpyHostToDevice);
        cudaMemcpy(U->CUDA_value, A->CUDA_value, A->nnz * sizeof(calculate_type), cudaMemcpyDeviceToDevice);
    }
    else if (A->nnz > 15000)//A->nnz > 1e4
    {
        pangulu_getrf_cuda_dense_kernel(A->row,
                                        A->rowpointer[A->row],
                                        U->CUDA_nnzU,
                                        A->CUDA_rowpointer,
                                        A->CUDA_columnindex,
                                        A->CUDA_value,
                                        L->CUDA_rowpointer,
                                        L->CUDA_columnindex,
                                        L->CUDA_value,
                                        U->CUDA_rowpointer,
                                        U->CUDA_columnindex,
                                        U->CUDA_value);
    }
    else
    {
        pangulu_getrf_cuda_kernel(A->row,
                                  A->rowpointer[A->row],
                                  U->CUDA_nnzU,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  A->CUDA_value,
                                  L->CUDA_rowpointer,
                                  L->CUDA_columnindex,
                                  L->CUDA_value,
                                  U->CUDA_rowpointer,
                                  U->CUDA_columnindex,
                                  U->CUDA_value);
    }
}

void pangulu_getrf_interface_G_V1(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{

#ifdef ADAPTIVE_KERNEL_SELECTION

    if (A->row == A->rowpointer[A->row])
    {
        calculate_type *tmp = (calculate_type *)malloc(sizeof(calculate_type) * A->rowpointer[A->row]);
        for (int i = 0; i < A->row; i++)
        {
            tmp[i] = 1.0;
        }
        cudaMemcpy(L->CUDA_value, tmp, A->rowpointer[A->row] * sizeof(calculate_type), cudaMemcpyHostToDevice);
        cudaMemcpy(U->CUDA_value, A->CUDA_value, A->rowpointer[A->row] * sizeof(calculate_type), cudaMemcpyDeviceToDevice);
    }
    else if (A->rowpointer[A->row] > 15000)//A->nnz > 1e4
    {
        pangulu_getrf_cuda_dense_kernel(A->row,
                                        A->rowpointer[A->row],
                                        U->CUDA_nnzU,
                                        A->CUDA_rowpointer,
                                        A->CUDA_columnindex,
                                        A->CUDA_value,
                                        L->CUDA_rowpointer,
                                        L->CUDA_columnindex,
                                        L->CUDA_value,
                                        U->CUDA_rowpointer,
                                        U->CUDA_columnindex,
                                        U->CUDA_value);
    }
    else
    {
        pangulu_getrf_cuda_kernel(A->row,
                                  A->rowpointer[A->row],
                                  U->CUDA_nnzU,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  A->CUDA_value,
                                  L->CUDA_rowpointer,
                                  L->CUDA_columnindex,
                                  L->CUDA_value,
                                  U->CUDA_rowpointer,
                                  U->CUDA_columnindex,
                                  U->CUDA_value);
    }

#else
    pangulu_getrf_cuda_kernel(A->row,
                                  A->rowpointer[A->row],
                                  U->CUDA_nnzU,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  A->CUDA_value,
                                  L->CUDA_rowpointer,
                                  L->CUDA_columnindex,
                                  L->CUDA_value,
                                  U->CUDA_rowpointer,
                                  U->CUDA_columnindex,
                                  U->CUDA_value);
#endif

    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, U);
    
}
void pangulu_getrf_interface_G_V2(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
    pangulu_getrf_cuda_dense_kernel(A->row,
                                    A->rowpointer[A->row],
                                    U->CUDA_nnzU,
                                    A->CUDA_rowpointer,
                                    A->CUDA_columnindex,
                                    A->CUDA_value,
                                    L->CUDA_rowpointer,
                                    L->CUDA_columnindex,
                                    L->CUDA_value,
                                    U->CUDA_rowpointer,
                                    U->CUDA_columnindex,
                                    U->CUDA_value);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(L, L);
    pangulu_Smatrix_CUDA_memcpy_value_CSC(U, U);
}


#endif
