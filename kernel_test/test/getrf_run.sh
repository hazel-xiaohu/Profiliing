export file_path=./test_matrices
GETRF_NUM=$(find -name "*getrf_A.cbd" | wc -l)
echo GETRF_NUM_SUM:${GETRF_NUM}
for ((i=0; i<${GETRF_NUM}; i++))
do
    export A_filename=${file_path}/getrf/${i}_getrf_A.cbd
    export L_filename=${file_path}/getrf/${i}_getrf_L.cbd
    export U_filename=${file_path}/getrf/${i}_getrf_U.cbd
    echo GETRF ${i}: 
    echo 
    ./getrf -A ${A_filename} -L ${L_filename} -U ${U_filename}
done