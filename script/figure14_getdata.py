import numpy as np
num = 16
matrix_name = ["apache2", "ASIC_680k", "audikw_1",
               "cage12", "CoupCons3D", "dielFilterV3real", "ecology1", "G3_circuit", "Ga41As41H72", "Hook_1498", "inline_1", "ldoor", "nlpkkt80", "Serena", "Si87H76", "SiO2"]
dataorigin = [0.0 for i in range(num)]
index = 0
with open('../data/test128_origin.out', 'r') as f:
    file_lines = f.readlines()
    words = [file_line.strip().split(' ') for file_line in file_lines]
    for word_ in words:
        if (len(word_) >= 6):
            if (word_[3] == "numerical"):
                dataorigin[index] = word_[6]
                index = index+1

dataselect = [0.0 for i in range(num)]
index = 0
with open('../data/test128_select.out', 'r') as f:
    file_lines = f.readlines()
    words = [file_line.strip().split(' ') for file_line in file_lines]
    for word_ in words:
        if (len(word_) >= 6):
            if (word_[3] == "numerical"):
                dataselect[index] = word_[6]
                index = index+1


data128 = [0.0 for i in range(num)]
index = 0
with open('../data/test128_full.out', 'r') as f:
    file_lines = f.readlines()
    words = [file_line.strip().split(' ') for file_line in file_lines]
    for word_ in words:
        if (len(word_) >= 6):
            if (word_[3] == "numerical"):
                data128[index] = word_[6]
                index = index+1

with open('../data/figure14_01.csv', 'w') as f:
    f.write("matrix,no_select,select,select_nosynchonrize\n")
    for i in range(0, 8):
        f.write(matrix_name[i]+","+dataorigin[i] +
                ","+dataselect[i]+","+data128[i]+"\n")

with open('../data/figure14_02.csv', 'w') as f:
    f.write("matrix,no_select,select,select_nosynchonrize\n")
    for i in range(8, 16):
        f.write(matrix_name[i]+","+dataorigin[i] +
                ","+dataselect[i]+","+data128[i]+"\n")
