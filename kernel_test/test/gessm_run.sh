export file_path=./test_matrices
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