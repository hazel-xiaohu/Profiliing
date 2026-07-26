export file_path=./test_matrices
TSTRF_NUM=$(find -name "*tstrf_B.cbd" | wc -l)
echo TSTRF_NUM_SUM:${TSTRF_NUM}
for ((i=0; i<${TSTRF_NUM}; i++))
do
    export B_filename=${file_path}/tstrf/${i}_tstrf_B.cbd
    export U_filename=${file_path}/tstrf/${i}_tstrf_U.cbd
    echo TSTRF ${i}: 
    echo 
    ./tstrf -B ${B_filename} -U ${U_filename}
done