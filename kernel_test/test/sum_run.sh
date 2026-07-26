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