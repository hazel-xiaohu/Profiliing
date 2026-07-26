import numpy as np
import pandas as pd
num = 16
matrix_name = ["apache2", "ASIC_680k", "audikw_1",
               "cage12", "CoupCons3D", "dielFilterV3real", "ecology1", "G3_circuit", "Ga41As41H72", "Hook_1498", "inline_1", "ldoor", "nlpkkt80", "Serena", "Si87H76", "SiO2"]
data128symbolic =  [0.0 for i in range(num)]
index=0

with open('../data/figure11.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=5):
            if(word_[1] == "symbolic"):
                data128symbolic[index]=word_[4]
                index=index+1

with open('../data/figure11_01.csv', 'w') as f:
    f.write("matrix,pangu1\n")
    for i in range(0, 8):
        f.write(matrix_name[i]+","+data128symbolic[i]+"\n")

with open('../data/figure11_02.csv', 'w') as f:
    f.write("matrix,pangu1\n")
    for i in range(8, 16):
        f.write(matrix_name[i]+","+data128symbolic[i]+"\n")


#f=open('')