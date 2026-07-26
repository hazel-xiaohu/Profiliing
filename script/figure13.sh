#get ld_library_path
export LD_LIBRARY_PATH=../lib/:${LD_LIBRARY_PATH}

#run figure13 script

cp ../src/common_versions/pangulu_common_select.h ../src/pangulu_common.h
cd ..
make
cd ./script
for matrix_name in {apache2,ASIC_680k,audikw_1,cage12,CoupCons3D,dielFilterV3real,ecology1,G3_circuit,Ga41As41H72,Hook_1498,inline_1,ldoor,nlpkkt80,Serena,Si87H76,SiO2}
do
    echo "Running matrix "${matrix_name} 
    bash run.sh ../matrix/${matrix_name}.mtx 128 >>../data/test128_select.out
done

cp ../src/common_versions/pangulu_common_full.h ../src/pangulu_common.h
cd ..
make
cd ./script
for matrix_name in {apache2,ASIC_680k,audikw_1,cage12,CoupCons3D,dielFilterV3real,ecology1,G3_circuit,Ga41As41H72,Hook_1498,inline_1,ldoor,nlpkkt80,Serena,Si87H76,SiO2}
do
    echo "Running matrix "${matrix_name} 
    bash run.sh ../matrix/${matrix_name}.mtx 128 >>../data/test128_full.out
done

python3 figure13_getdata.py
echo "Begin painting figure13"
python3 figure13.py ../data/figure13_01.csv ../data/superlu_data/figure13_01.csv ../data/figure13_02.csv ../data/superlu_data/figure13_02.csv
echo "End painting figure13"


