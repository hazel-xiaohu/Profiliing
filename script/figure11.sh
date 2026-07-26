#get ld_library_path
export LD_LIBRARY_PATH=../lib/:${LD_LIBRARY_PATH}

#run figure11 script
#you can make and run without any changes

cp ../src/common_versions/pangulu_common_full.h ../src/pangulu_common.h
cd ..
make
cd ./script

rm -rf ../data/figure11.out

for matrix_name in {apache2,ASIC_680k,audikw_1,cage12,CoupCons3D,dielFilterV3real,ecology1,G3_circuit,Ga41As41H72,Hook_1498,inline_1,ldoor,nlpkkt80,Serena,Si87H76,SiO2}
do
    echo "Running matrix "${matrix_name} 

    bash run.sh ../matrix/${matrix_name}.mtx 128 >>../data/figure11.out
done


python3 figure11_getdata.py

echo "Begin painting figure11"
python3 figure11.py ../data/figure11_01.csv ../data/superlu_data/figure11_01.csv ../data/figure11_02.csv ../data/superlu_data/figure11_02.csv
echo "End painting figure11"
