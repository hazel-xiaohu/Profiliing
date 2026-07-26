#CUDA_PATH = /usr/local/cuda
LU_INC =-I../include
#CUDA_INC = -I$(CUDA_PATH)/include
#CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse

#NVCC = nvcc
CC = gcc 
CPP = g++
MPICPP = mpic++

CPPFLAGS = -fopenmp -O3 
MPICPPFLAGS = -O3 -Wall -std=c++11 -lm -lpthread -fopenmp $(LU_INC)
#NVCCFLAGS = -O3 -w -Xptxas -dlcm=cg -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 $(LU_INC) $(CUDA_INC) $(CUDA_LIB)
MPICPPLINK = -L/mpich/lib 

METISFLAGS = /home/huz/PanguLU-main/metis/build/Linux-x86_64/libmetis/libmetis.a

#gpu=1
