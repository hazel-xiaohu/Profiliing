import numpy as np
num = 16
matrix_name = ["apache2", "ASIC_680k", "audikw_1",
               "cage12", "CoupCons3D", "dielFilterV3real", "ecology1",
               "G3_circuit", "Ga41As41H72", "Hook_1498", "inline_1", 
               "ldoor", "nlpkkt80", "Serena", "Si87H76", "SiO2"]
data1 =  [0.0 for i in range(num)]
index=0
with open('../data/test1sum.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                data1[index]=word_[8]
                index=index+1

data2 =  [0.0 for i in range(num)]
index=0
with open('../data/test2sum.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                data2[index]=word_[8]
                index=index+1

data4 =  [0.0 for i in range(num)]
index=0
with open('../data/test4sum.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                data4[index]=word_[8]
                index=index+1

data8 =  [0.0 for i in range(num)]
index=0
with open('../data/test8sum.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                data8[index]=word_[8]
                index=index+1


data16 =  [0.0 for i in range(num)]
index=0
with open('../data/test16sum.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                data16[index]=word_[8]
                index=index+1

data32 =  [0.0 for i in range(num)]
index=0
with open('../data/test32sum.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                data32[index]=word_[8]
                index=index+1

data64 =  [0.0 for i in range(num)]
index=0
with open('../data/test64sum.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                data64[index]=word_[8]
                index=index+1

data128 =  [0.0 for i in range(num)]
index=0
with open('../data/test128sum.out','r') as f:
     file_lines = f.readlines()
     words = [file_line.strip().split(' ') for file_line in file_lines]
     for word_ in words:
        if(len(word_)>=6):
            if(word_[3] == "numerical"):
                data128[index]=word_[8]
                index=index+1

#for i in range(16):
#    print(data1[i],data2[i],data4[i],data8[i],data16[i],data32[i],data64[i],data128[i])

with open('../data/figure12.csv', 'w') as f:
    f.write("matrix,pangu-1,pangu-2,pangu-4,pangu-8,pangu-16,pangu-32,pangu-64,Pangu-128\n")
    for i in range(0, 16):
        f.write(matrix_name[i]+","+str(data1[i]) +
                ","+data2[i]+","+data4[i]+","+
                data8[i]+","+data16[i]+","+data32[i]+","
                +data64[i]+","+data128[i]+"\n")