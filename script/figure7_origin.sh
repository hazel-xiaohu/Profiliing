#get ld_library_path
export LD_LIBRARY_PATH=../lib/:${LD_LIBRARY_PATH}
cp ../src/common_versions/pangulu_common_output_matrices.h ../src/pangulu_common.h
cd ..
echo "Compiling"
make 
echo "0 0 0 0" >./kernel_test/test/test_matrices/matrices_number.txt
echo "Deleting remaining matrix blocks"
rm -rf ./kernel_test/test/csv_out/*csv
for name in {getrf,gessm,tstrf,ssssm}
do
    rm -rf ./kernel_test/test/test_matrices/${name}
    mkdir ./kernel_test/test/test_matrices/${name}
done
echo "Deleted remaining matrix blocks"
export LD_LIBRARY_PATH=./test/:${LD_LIBRARY_PATH}
echo "Exporting matrix blocks from 16 matrices"
export OMP_NUM_THREADS=4
for matrix_name in {apache2,ASIC_680k,audikw_1,cage12,CoupCons3D,dielFilterV3real,ecology1,G3_circuit,Ga41As41H72,Hook_1498,inline_1,ldoor,nlpkkt80,Serena,Si87H76,SiO2}
do
    echo "Exporting matrix blocks from " ${matrix_name}
    mpirun -np 1 ./test/PanguLU -F ./matrix/${matrix_name}.mtx 
    echo "Exported matrix blocks from " ${matrix_name}
done


#run figure7 script
#you can make and run without any changes

cd ./kernel_test/test/
export file_path=./test_matrices
GETRF_NUM=$(find -name "*getrf_A.cbd" | wc -l)
echo GETRF_NUM_SUM:${GETRF_NUM}
echo "n,nnz,max_row_nnz,min_row_nnz,max_col_nnz,min_col_nnz,average_row_nnz,average_col_nnz,variety_row,variety_col,cpu_dense,gpu_dense,gpu_sparse" >./csv_out/figure7_GETRF.csv
for ((i=0; i<${GETRF_NUM}; i++))
do
    export A_filename=${file_path}/getrf/${i}_getrf_A.cbd
    export L_filename=${file_path}/getrf/${i}_getrf_L.cbd
    export U_filename=${file_path}/getrf/${i}_getrf_U.cbd
    echo GETRF ${i}: 
    echo 
    ./getrf -A ${A_filename} -L ${L_filename} -U ${U_filename}
done

echo "n,nnz_B,max_row_nnz_B,min_row_nnz_B,max_col_nnz_B,min_col_nnz_B,average_row_nnz_B,average_col_nnz_B,variety_row_B,variety_col_B,nnz_L,max_row_nnz_L,min_row_nnz_L,max_col_nnz_L,min_col_nnz_L,average_row_nnz_L,average_col_nnz_L,variety_row_L,variety_col_L,cpu_1,cpu_2,cpu_3,cpu_4,cpu_5,cpu_6,gpu_7,gpu_8,gpu_9,gpu_10,gpu_11" >./csv_out/figure7_GESSM.csv
GESSM_NUM=$(find -name "*gessm_B.cbd" | wc -l)
echo GESSM_NUM_SUM:${GESSM_NUM}
for ((i=0; i<${GESSM_NUM}; i++))
do
    export B_filename=${file_path}/gessm/${i}_gessm_B.cbd
    export L_filename=${file_path}/gessm/${i}_gessm_L.cbd
    echo GESSM ${i}: 
    echo 
    ./gessm -B ${B_filename} -L ${L_filename}
done

TSTRF_NUM=$(find -name "*tstrf_B.cbd" | wc -l)
echo "n,nnz_B,max_row_nnz_B,min_row_nnz_B,max_col_nnz_B,min_col_nnz_B,average_row_nnz_B,average_col_nnz_B,variety_row_B,variety_col_B,nnz_U,max_row_nnz_U,min_row_nnz_U,max_col_nnz_U,min_col_nnz_U,average_row_nnz_U,average_col_nnz_U,variety_row_U,variety_col_U,cpu_1,cpu_2,cpu_3,cpu_4,cpu_5,cpu_6,gpu_7,gpu_8,gpu_9,gpu_10,gpu_11" >./csv_out/figure7_TSTRF.csv
echo TSTRF_NUM_SUM:${TSTRF_NUM}
for ((i=0; i<${TSTRF_NUM}; i++))
do
    export B_filename=${file_path}/tstrf/${i}_tstrf_B.cbd
    export U_filename=${file_path}/tstrf/${i}_tstrf_U.cbd
    echo TSTRF ${i}: 
    echo 
    ./tstrf -B ${B_filename} -U ${U_filename}
done

SSSSM_NUM=$(find -name "*ssssm_B.cbd" | wc -l)
echo "n,nnz_A,nnz_B,nnz_C,calcalution_sum,max_row_nnz_C,min_row_nnz_C,max_col_nnz_C,min_col_nnz_C,average_row_nnz_C,average_col_nnz_C,variety_row,variety_col,cpu_dense,cpu_thmkl,gpu_dense,gpu_sparse" >./csv_out/figure7_SSSSM.csv
echo SSSSM_NUM_SUM:${SSSSM_NUM}
for ((i=0; i<${SSSSM_NUM}; i++))
do
    export A_filename=${file_path}/ssssm/${i}_ssssm_A.cbd
    export B_filename=${file_path}/ssssm/${i}_ssssm_B.cbd
    export C_filename=${file_path}/ssssm/${i}_ssssm_C.cbd
    echo SSSSM ${i}: 
    echo 
    ./ssssm -A ${A_filename} -B ${B_filename} -C ${C_filename}
done
cd ../../
cp ./kernel_test/test/csv_out/*csv ./data
cd ./script
echo "Begin painting figure7"
python3 figure7.py ../data/figure7_GESSM.csv ../data/figure7_TSTRF.csv ../data/figure7_SSSSM.csv ../data/figure7_GETRF.csv
echo "End painting figure7"
