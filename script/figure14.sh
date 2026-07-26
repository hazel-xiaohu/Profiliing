#get ld_library_path
export LD_LIBRARY_PATH=../lib/:${LD_LIBRARY_PATH}

#run figure14 script
cp ../src/common_versions/pangulu_common_origin.h ../src/pangulu_common.h
cd ..
make
cd ./script
for matrix_name in {apache2,ASIC_680k,audikw_1,cage12,CoupCons3D,dielFilterV3real,ecology1,G3_circuit,Ga41As41H72,Hook_1498,inline_1,ldoor,nlpkkt80,Serena,Si87H76,SiO2}
do
    echo "Running matrix "${matrix_name} 
    bash run.sh ../matrix/${matrix_name}.mtx 128 >>../data/test128_origin.out
done

# you should run figure13.sh first

python3 figure14_getdata.py
echo "Begin painting figure14"
python3 figure14.py ../data/figure14_01.csv ../data/figure14_02.csv 
echo "End painting figure14"


