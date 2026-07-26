#include "pangulu_common.h"

#include "pangulu_test.h"
#include "pangulu_calib.h"

int_t CPU_MEMORY=0;
int_t GPU_MEMORY=0;
int_t HEAP_SELECT;
calculate_type *TEMP_A_value = NULL;
idx_int *CUDA_B_idx_COL = NULL;
calculate_type *CUDA_TEMP_value = NULL;
int_t *ssssm_col_ops_u = NULL;
idx_int *ssssm_ops_pointer = NULL;
idx_int *getrf_diagIndex_csc = NULL;
idx_int *getrf_diagIndex_csr = NULL;
int_t SYMBOLIC_NNZ=0;
int_t STREAM_DENSE_INDEX=0;
int_t INDEX_NUM=0;
idx_int PANGU_OMP_NUM_THREADS=1;

int_t GPU_MEMORY_FLAG=0;

int_t FLOP=0;
double TIME_transport = 0.0;
double TIME_isend = 0.0;
double TIME_receive = 0.0;
double TIME_getrf = 0.0;
double TIME_tstrf = 0.0;
double TIME_gessm = 0.0;
double TIME_gessm_dense = 0.0;
double TIME_gessm_sparse = 0.0;
double TIME_ssssm = 0.0;
double TIME_cuda_memcpy = 0.0;
double TIME_wait = 0.0;
double calculate_TIME_wait = 0.0;

char *block_matrix_name;

int_t calculate_time=0;

int_32t RANK;
int_32t LEVEL;
int_32t OMP_THREAD;

#ifdef OUTPUT_MATRICES
int ssssm_number = 0;
int getrf_number = 0;
int gessm_number = 0;
int tstrf_number = 0;
#endif

void pangulu(int ARGC, char **ARGV)
{
    int_32t rank;
    int_32t size;
#ifndef OVERLAP
    MPI_Init(&ARGC, &ARGV);
#else
    int provided;
    MPI_Init_thread(&ARGC,&ARGV,MPI_THREAD_MULTIPLE,&provided);
#endif

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    pangulu_common *common = (pangulu_common *)pangulu_malloc(sizeof(pangulu_common));
    common->rank = rank;
    common->size = size;
    
    RANK=rank;

    pangulu_get_common(common, ARGC, ARGV,size);
    pangulu_Smatrix *oringin_Smatrix = (pangulu_Smatrix *)pangulu_malloc(sizeof(pangulu_Smatrix));
    pangulu_init_pangulu_Smatrix(oringin_Smatrix);
    if (rank == 0)
    {
#ifdef OUTPUT_MATRICES
        char MATRICES_NUMBER_FILE[1024];
        sprintf(MATRICES_NUMBER_FILE, "%s/matrices_number.txt", OUTPUT_FILE);
        FILE *fp = fopen(MATRICES_NUMBER_FILE, "r");
        fscanf(fp, "%d %d %d %d", &ssssm_number, &getrf_number, &gessm_number, &tstrf_number);
        fclose(fp);
#endif
    }
    if (rank == 0)
    {
        int_32t tag=0;
#ifdef READ_BINARY
        pangulu_read_pangulu_Smatrix(oringin_Smatrix, common->file_name);
        tag++;
        
#endif
        

#ifdef METIS_ANALYSIS
        pangulu_metis_interface(oringin_Smatrix,common->file_name);
        tag++;
#endif


        if(oringin_Smatrix->row==0){
            printf("matrix read error row is 0\n");
            return ;
        }
        if(tag==1){
            //printf("read right matrix name %s\n",common->file_name);
        }
        else{
            printf("error analysis %d\n",tag);
        }
        //pangulu_display_pangulu_Smatrix(oringin_Smatrix);
    }
    //return ;
    pangulu_test(common, oringin_Smatrix);
    
#ifdef CHECK_ANS
    if (rank == 0)
    {
        pangulu_destroy_part_pangulu_Smatrix(oringin_Smatrix);
    }
    else{
        free(oringin_Smatrix);
        oringin_Smatrix=NULL;
    }  
#endif


    if (rank == 0)
    {
#ifdef OUTPUT_MATRICES
        char MATRICES_NUMBER_FILE[1024];
        sprintf(MATRICES_NUMBER_FILE, "%s/matrices_number.txt", OUTPUT_FILE);
        FILE *fp = fopen(MATRICES_NUMBER_FILE, "w");
        fprintf(fp, "%d %d %d %d", ssssm_number, getrf_number, gessm_number, tstrf_number);
        fclose(fp);
#endif
    }

    if(rank!=0)
    pangulu_destroy_pangulu_common(common);
    
    pangulu_calib_close_catalog();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return ;
}
