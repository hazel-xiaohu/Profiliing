#ifndef PANGULU_SSSSM_FP64_CUDA_H
#define PANGULU_SSSSM_FP64_CUDA_H

#include "pangulu_common.h"
#include "pangulu_cuda.h"

static int pangulu_ssssm_use_dense_kernel(pangulu_Smatrix *A)
{
    const char *mode = getenv("PANGULU_SSSSM_KERNEL");
    if (mode != NULL)
    {
        if (strcmp(mode, "sparse") == 0)
        {
            return 0;
        }
        if (strcmp(mode, "dense") == 0)
        {
            return 1;
        }
    }

    int n = A->row;
    int nnz_a = A->columnpointer[n] - A->columnpointer[0];
    double sparsity_A = (double)nnz_a / (double)(n * n);
    return sparsity_A >= 0.001;
}

static int pangulu_ssssm_sparse_launch_count(pangulu_Smatrix *A)
{
#ifdef ADAPTIVE_KERNEL_SELECTION
    int launches = 0;
    for (int layer = 1; layer <= 10; ++layer)
    {
        if (A->bin_rowpointer[layer + 1] > A->bin_rowpointer[layer])
        {
            launches += 1;
        }
    }
    return launches;
#else
    return (A->bin_rowpointer[2] > A->bin_rowpointer[1]) ? 1 : 0;
#endif
}

void pangulu_ssssm_fp64_cuda(pangulu_Smatrix *A,
                                 pangulu_Smatrix *L,
                                 pangulu_Smatrix *U)
    {
#ifdef ADAPTIVE_KERNEL_SELECTION

    if(!pangulu_ssssm_use_dense_kernel(A)){
     pangulu_calib_cuda_event calib_event;
     pangulu_calib_kernel_event_begin(&calib_event);
     pangulu_ssssm_cuda_kernel(A->row,
                                  A->bin_rowpointer,
                                  A->CUDA_bin_rowpointer,
                                  A->CUDA_bin_rowindex,
                                  U->CUDA_rowpointer,
                                  U->CUDA_columnindex,
                                  U->CUDA_value,
                                  L->CUDA_rowpointer,
                                  L->CUDA_columnindex,
                                  L->CUDA_value,
                                  A->CUDA_rowpointer,
                                  A->CUDA_columnindex,
                                  A->CUDA_value);
     pangulu_calib_kernel_event_end(&calib_event, pangulu_ssssm_sparse_launch_count(A));
    }
    else{
    pangulu_calib_cuda_event calib_event;
    pangulu_calib_kernel_event_begin(&calib_event);
    pangulu_ssssm_dense_cuda_kernel(A->row,
                                    A->columnpointer[A->row],
                                    U->columnpointer[U->row],
                                    L->CUDA_rowpointer,
                                    L->CUDA_columnindex,
                                    L->CUDA_value,
                                    U->CUDA_rowpointer,
                                    U->CUDA_columnindex,
                                    U->CUDA_value,
                                    A->CUDA_rowpointer,
                                    A->CUDA_columnindex,
                                    A->CUDA_value);
    pangulu_calib_kernel_event_end(&calib_event, 4);
    }
#else
        pangulu_calib_cuda_event calib_event;
        pangulu_calib_kernel_event_begin(&calib_event);
        pangulu_ssssm_cuda_kernel(A->row,
                              A->bin_rowpointer,
                              A->CUDA_bin_rowpointer,
                              A->CUDA_bin_rowindex,
                              U->CUDA_rowpointer,
                              U->CUDA_columnindex,
                              U->CUDA_value,
                              L->CUDA_rowpointer,
                              L->CUDA_columnindex,
                              L->CUDA_value,
                              A->CUDA_rowpointer,
                              A->CUDA_columnindex,
                              A->CUDA_value);
        pangulu_calib_kernel_event_end(&calib_event, pangulu_ssssm_sparse_launch_count(A));
#endif


    }


void pangulu_ssssm_interface_G_V1(pangulu_Smatrix *A,
                                    pangulu_Smatrix *L,
                                    pangulu_Smatrix *U)
{
    pangulu_calib_cuda_event calib_event;
    pangulu_calib_kernel_event_begin(&calib_event);
    pangulu_ssssm_cuda_kernel(A->row,
                              A->bin_rowpointer,
                              A->CUDA_bin_rowpointer,
                              A->CUDA_bin_rowindex,
                              U->CUDA_rowpointer,
                              U->CUDA_columnindex,
                              U->CUDA_value,
                              L->CUDA_rowpointer,
                              L->CUDA_columnindex,
                              L->CUDA_value,
                              A->CUDA_rowpointer,
                              A->CUDA_columnindex,
                              A->CUDA_value);
    pangulu_calib_kernel_event_end(&calib_event, pangulu_ssssm_sparse_launch_count(A));
}
void pangulu_ssssm_interface_G_V2(pangulu_Smatrix *A,
                                  pangulu_Smatrix *L,
                                  pangulu_Smatrix *U)
{
    pangulu_calib_cuda_event calib_event;
    pangulu_calib_kernel_event_begin(&calib_event);
    pangulu_ssssm_dense_cuda_kernel(A->row,
                                    A->columnpointer[A->row],
                                    U->columnpointer[U->row],
                                    L->CUDA_rowpointer,
                                    L->CUDA_columnindex,
                                    L->CUDA_value,
                                    U->CUDA_rowpointer,
                                    U->CUDA_columnindex,
                                    U->CUDA_value,
                                    A->CUDA_rowpointer,
                                    A->CUDA_columnindex,
                                    A->CUDA_value);
    pangulu_calib_kernel_event_end(&calib_event, 4);
}
#endif
