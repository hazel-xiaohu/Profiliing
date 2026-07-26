export file_path=./test_matrices
SSSSM_NUM=$(find -name "*ssssm_B.cbd" | wc -l)
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