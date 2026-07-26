import numpy as np
num = 16
matrix_name = ["apache2", "ASIC_680k", "audikw_1",
               "cage12", "CoupCons3D", "dielFilterV3real", "ecology1", "G3_circuit", "Ga41As41H72", "Hook_1498", "inline_1", "ldoor", "nlpkkt80", "Serena", "Si87H76", "SiO2"]

dataorigin =  [0.0 for i in range(num)]
index=0
with open('../data/test128_full.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                dataorigin[index]=word_[6]
                index=index+1

dataselect =  [0.0 for i in range(num)]
index=0
with open('../data/test128_select.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                dataselect[index]=word_[6]
                index=index+1

calculate128 =  [0.0 for i in range(num)]
index=-1
with open('../data/test128_full.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split('\t') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[0] == "rank"):
                index=index+1
            elif(len(word_)>=6):
                calculate128[index]+=float(word_[5])
    


# for i in range(0,8):
#     print(float(dataorigin[i])-calculate128[i]/128.0,float(dataselect[i])-calculate128[i]/128.0)


with open('../data/figure13_01.csv', 'w') as f:
    f.write("matrix,pangu_commu_syn,pangu_commu_syn_free\n")
    for i in range(0, 8):
        commu_syn=float(dataorigin[i])-calculate128[i]/128.0
        commu_syn_free=float(dataselect[i])-calculate128[i]/128.0
        f.write(matrix_name[i]+","+str(commu_syn)+","+str(commu_syn_free)+"\n")

with open('../data/figure13_02.csv', 'w') as f:
    f.write("matrix,pangu_commu_syn,pangu_commu_syn_free\n")
    for i in range(8, 16):
        commu_syn=float(dataorigin[i])-calculate128[i]/128.0
        commu_syn_free=float(dataselect[i])-calculate128[i]/128.0
        f.write(matrix_name[i]+","+str(commu_syn)+","+str(commu_syn_free)+"\n")
#f=open('')