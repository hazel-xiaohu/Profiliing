#CUDA_PATH = /openbayes/home/cuda-12.2/targets/x86_64-linux
CUDA_PATH = /usr/local/cuda-12.1
#CUDA_PATH = /openbayes/home/cuda-11.3
LU_INC = -I../include
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -lcudart -lcusparse

NVCC = nvcc
CC = gcc
CPP = g++
MPICPP = mpic++

CPPFLAGS = -fopenmp -O3
MPICPPFLAGS = -O3 -Wall -std=c++11 -lm -lpthread -fopenmp $(LU_INC) $(CUDA_INC)
NVCCFLAGS = -O3 -w -Xptxas -dlcm=cg -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90 $(LU_INC) $(CUDA_INC) $(CUDA_LIB) -I/openbayes/home/cuda/include -allow-unsupported-compiler
#NVCCFLAGS = -O3 -w -Xptxas -dlcm=cg -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_120,code=compute_120 $(LU_INC) $(CUDA_INC) $(CUDA_LIB) -I/openbayes/home/cuda/include -allow-unsupported-compiler
#MPICPPLINK = -L/data/app/mpi/openmpi/4.1.4-gcc/lib
#MPICPPLINK = -L/openbayes/home/mpich-install/lib/

METISFLAGS = /openbayes/home/PanguLU-main-bak/metis/build/Linux-x86_64/libmetis/libmetis.a

gpu=1
