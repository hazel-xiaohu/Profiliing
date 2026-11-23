#ifndef PANGULU_TEST_H
#define PANGULU_TEST_H


#include "pangulu_common.h"
#include "pangulu_utils.h"
#include "pangulu_preprocess.h"


#include "pangulu_numerical.h"

#include "pangulu_check.h"
#include "pangulu_destroy.h"
#include "pangulu_sptrsv.h"


    void pangulu_test(pangulu_common *common,
                      pangulu_Smatrix *oringin_Smatrix)
    {



        
        int_32t rank = common->rank;
        int_32t P = common->P;
        int_32t Q = common->Q;

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);
        common->N = pangulu_Bcast_N(oringin_Smatrix->row, 0);
        int_t N = common->N;
        if (rank == 0)
        {
#ifdef ADAPTIVE_KERNEL_SELECTION
        printf("ADAPTIVE_KERNEL_SELECTION -------------ON\n");
#else
        printf("ADAPTIVE_KERNEL_SELECTION ------------OFF\n");
#endif
#ifdef SYNCHRONIZE_FREE
        printf("SYNCHRONIZE_FREE ----------------------ON\n");
#else
        printf("SYNCHRONIZE_FREE ---------------------OFF\n");
#endif
                printf("N is %ld ,NNZ is %ld\n", N, oringin_Smatrix->rowpointer[N]);
                OMP_THREAD = common->omp_thread;
                // printf("OMP_THREAD: %d \n",OMP_THREAD);
        }

#ifdef GPU_OPEN
        pangulu_cuda_device_init(rank);
#endif

        

        pangulu_block_Smatrix *block_Smatrix=(pangulu_block_Smatrix *)pangulu_malloc(sizeof(pangulu_block_Smatrix));
        pangulu_block_common *block_common=(pangulu_block_common *)pangulu_malloc(sizeof(pangulu_block_common));

#ifdef SYMBOLIC
        if(rank==0){
                pangulu_symbolic(oringin_Smatrix,block_Smatrix);
//                common->NB=pangulu_select_NB(N,SYMBOLIC_NNZ);
        }
#endif

        
        common->NB = pangulu_Bcast_N(common->NB, 0);
        int_32t NB = common->NB;

#ifdef READ_BINARY_BLOCK

    if(rank!=0){
        int_t block_matrix_length = strlen(block_matrix_name);
        char matrix_location[10] = "_00000/";
        int_t now = 10000;
        int_t now_NB = NB;
        for (int_t i = 0; i < 4; i++)
        {
                matrix_location[i + 1] = (now_NB / now) + '0';
                now_NB = now_NB % now;
                now /= 10;
        }
        strcpy(block_matrix_name + block_matrix_length - 4, matrix_location);
    }
    
#endif

        block_common->rank=rank;
        block_common->P=P;
        block_common->Q=Q;
        block_common->NB=NB;
        block_common->N=N;
        block_common->block_length = pangulu_Calculate_Block(N, NB);
        block_common->sum_rank_size=common->sum_rank_size;
        block_common->max_PQ=PANGULU_MAX(P,Q);

        block_common->every_level_length=block_common->block_length;   


#ifdef SYNCHRONIZE_FREE
        block_common->every_level_length=20;
#else
        block_common->every_level_length=1;
#endif
        pangulu_init_heap_select(0);


        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);
        pangulu_preprocess(block_common,
                           oringin_Smatrix,
                           block_Smatrix);
        
        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_stop(common);
        
        
        if (rank==0)
        {
            printf("rank %d the preprocess time is %lf s\n", rank,pangulu_get_spend_time(common));
        }
        N = block_common->N;
        //return ;

        //printf("rank %d GPU MEMORY:%ld\n",rank,GPU_MEMORY);

        block_common->rank_row_length = (block_common->block_length / P + (((block_common->block_length % P) > (rank / Q)) ? 1 : 0));
        block_common->rank_col_length = (block_common->block_length / Q + (((block_common->block_length % Q) > (rank % Q)) ? 1 : 0));
        
        calculate_time=0;
        

#ifndef PANGULU_FEATURE_REPLAY
        int_t vector_length = N;
        pangulu_vector *X_vector=(pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
        pangulu_vector *B_vector=(pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
        pangulu_vector *answer_vector=(pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
        
        if(rank==0){           
            X_vector=(pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
            B_vector=(pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));
            answer_vector=(pangulu_vector *)pangulu_malloc(sizeof(pangulu_vector));

            pangulu_read_pangulu_vector(X_vector, vector_length);
            pangulu_init_pangulu_vector(B_vector,vector_length);
            pangulu_init_pangulu_vector(answer_vector,vector_length);
            pangulu_pangulu_Smatrix_multiple_pangulu_vector_CSR(oringin_Smatrix, X_vector, B_vector);
            //pangulu_read_pangulu_vector(B_vector, vector_length);
            
        }
#endif
        
        block_common->every_level_length=PANGULU_MIN(block_common->every_level_length,block_common->block_length);
        
        

        //pangulu_sptrsv_preprocess(block_common,block_Smatrix,B_vector);        
#ifdef CHECK_TIME
        pangulu_time_init();
#endif

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef OVERLAP
        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);

        pangulu_create_pthread(block_common,
                               block_Smatrix);

        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_stop(common);
#endif        
        
        
        pangulu_time_init();
        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);


        pangulu_numerical(block_common,
                          block_Smatrix);


        pangulu_time_stop(common);
        MPI_Barrier(MPI_COMM_WORLD);
        
        if(rank==0){
                
                int_t another_calculate_time=0;
                for(int_t i=1;i<block_common->sum_rank_size;i++){
                        pangulu_recv_vector_int(&another_calculate_time,1,i,0);
                        calculate_time+=another_calculate_time;
                }
                //printf("rank %d calculate time %ld\n",rank,calculate_time*2);
                FLOP=calculate_time*2;
        }
        else{
                pangulu_send_vector_int(&calculate_time,1,0,0);
        }

        if (rank==0)
        {
#ifdef PANGULU_FEATURE_REPLAY
            printf("rank %d feature replay scheduling time is %lf s (target kernels skipped)\n",
                   rank, pangulu_get_spend_time(common));
#else
            printf("rank %d the numerical time is %lf s %lf GFLOPs\n", rank,pangulu_get_spend_time(common),FLOP/pangulu_get_spend_time(common)/1000000000.0);
#endif
        }
        
        

        //return ;
        fflush(NULL);
        MPI_Barrier(MPI_COMM_WORLD);
#ifdef CHECK_ANS
        pangulu_check(block_common,block_Smatrix,oringin_Smatrix);
        MPI_Barrier(MPI_COMM_WORLD);
        pangulu_time_start(common);
        
        //pangulu_sptrsv_L(block_common,block_Smatrix);

        //pangulu_init_heap_select(4);
        //pangulu_sptrsv_U(block_common,block_Smatrix);

        pangulu_time_stop(common);
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank==0)
        {
            //printf("rank %d the sptrsv time is %lf s\n", rank,pangulu_get_spend_time(common));
            
        }

        //pangulu_sptrsv_vector_gather(block_common,block_Smatrix,answer_vector);

        if(rank==0){
            //pangulu_display_pangulu_vector(X_vector);
            //pangulu_display_pangulu_vector(answer_vector);
            //pangulu_check_answer_vec2norm(X_vector, answer_vector, N); 
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        MPI_Barrier(MPI_COMM_WORLD);


#ifdef CHECK_TIME
        if(rank==0)
        pangulu_time_output();
        fflush(NULL);
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(NULL);
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(NULL);
        pangulu_time_simple_output(rank);
        
#endif
        pangulu_destroy(block_common,block_Smatrix);
        if(rank == 0){
            printf("pangulu_test---------------------------finish\n");
        }
        free(block_Smatrix);
        block_Smatrix=NULL;
        return;
    }


#endif
