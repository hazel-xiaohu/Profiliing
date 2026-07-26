# About more detail , you can read README
cd ./metis
rm -r metis
mkdir metis
make config prefix=./metis
make -j8 && make install
cd ..
make 
pip3 install numpy
pip3 install matplotlib
pip3 install pandas
