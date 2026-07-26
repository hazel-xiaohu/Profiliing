#get ld_library_path
export LD_LIBRARY_PATH=../lib/:${LD_LIBRARY_PATH}

#run figure12 script

cp ../src/common_versions/pangulu_common_full.h ../src/pangulu_common.h
cd ..
make
cd ./script

for mpi_number in {8,16,32,64,128}
do
    rm -rf ../data/test${mpi_number}sum.out        
done



for mpi_number in {8,16,32,64,128}
do
    for matrix_name in {apache2,ASIC_680k,audikw_1,cage12,CoupCons3D,dielFilterV3real,ecology1,G3_circuit,Ga41As41H72,Hook_1498,inline_1,ldoor,nlpkkt80,Serena,Si87H76,SiO2}
    do
        echo "Running matrix "${matrix_name} "with mpi_number" ${mpi_number}
        bash run.sh ../matrix/${matrix_name}.mtx ${mpi_number} >>../data/test${mpi_number}sum.out
    done
done

python3 figure12_getdata.py
echo "Begin painting figure12"
python3 figure12.py ../data/figure12.csv ../data/superlu_data/figure12.csv
echo "End painting figure12"

