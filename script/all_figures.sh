echo "Begin painting figure7"
python3 figure7.py ../data/pangulu_origin_data/figure7_GESSM.csv ../data/pangulu_origin_data/figure7_TSTRF.csv ../data/pangulu_origin_data/figure7_SSSSM.csv ../data/pangulu_origin_data/figure7_GETRF.csv
echo "End painting figure7"

echo "Begin painting figure11"
python3 figure11.py ../data/pangulu_origin_data/figure11_01.csv ../data/superlu_data/figure11_01.csv ../data/pangulu_origin_data/figure11_02.csv ../data/superlu_data/figure11_02.csv
echo "End painting figure11"

echo "Begin painting figure12"
python3 figure12.py ../data/pangulu_origin_data/figure12.csv ../data/superlu_data/figure12.csv
echo "End painting figure12"

echo "Begin painting figure13"
python3 figure13.py ../data/pangulu_origin_data/figure13_01.csv ../data/superlu_data/figure13_01.csv ../data/pangulu_origin_data/figure13_02.csv ../data/superlu_data/figure13_02.csv
echo "End painting figure13"

echo "Begin painting figure14"
python3 figure14.py ../data/pangulu_origin_data/figure14_01.csv ../data/pangulu_origin_data/figure14_02.csv 
echo "End painting figure14"
