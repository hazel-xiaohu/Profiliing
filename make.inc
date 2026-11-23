# GPU/toolchain contract. Do not decouple these settings: profiling labels are
# valid only for the exact architecture/toolchain pair used for model training.
#
#   make GPU_ARCH=sm_90   # H100, CUDA 12.1, PTX 8.1
#   make GPU_ARCH=sm_120  # RTX Pro 6000 Blackwell, CUDA 12.8, PTX 8.7
GPU_ARCH ?= sm_90

ifeq ($(GPU_ARCH),sm_90)
override CUDA_PATH := /usr/local/cuda
CUDA_EXPECTED_RELEASE := release 12.1
CUDA_ARCH_NUMBER := 90
else ifeq ($(GPU_ARCH),sm_120)
override CUDA_PATH := /output/toolchains/cuda-12.8
CUDA_EXPECTED_RELEASE := release 12.8
CUDA_ARCH_NUMBER := 120
else
$(error unsupported GPU_ARCH=$(GPU_ARCH); expected sm_90 or sm_120)
endif

override NVCC := $(CUDA_PATH)/bin/nvcc
CUDA_NVCC_VERSION := $(shell $(NVCC) --version 2>/dev/null)
ifeq ($(findstring $(CUDA_EXPECTED_RELEASE),$(CUDA_NVCC_VERSION)),)
$(error GPU_ARCH=$(GPU_ARCH) requires CUDA $(CUDA_EXPECTED_RELEASE), but $(NVCC) reports: $(CUDA_NVCC_VERSION))
endif
LU_INC = -I../include
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64 -Wl,-rpath,$(CUDA_PATH)/lib64 -lcudart

CC = gcc
CPP = g++
MPICPP = /openbayes/home/mpich-install/bin/mpic++

CPPFLAGS = -fopenmp -O3
MPICPPFLAGS = -O3 -Wall -std=c++11 -lm -lpthread -fopenmp $(LU_INC) $(CUDA_INC)
NVCCFLAGS = -O3 -w -Xptxas -dlcm=cg \
	-gencode=arch=compute_$(CUDA_ARCH_NUMBER),code=sm_$(CUDA_ARCH_NUMBER) \
	-gencode=arch=compute_$(CUDA_ARCH_NUMBER),code=compute_$(CUDA_ARCH_NUMBER) \
	$(LU_INC) $(CUDA_INC) -allow-unsupported-compiler
#MPICPPLINK = -L/data/app/mpi/openmpi/4.1.4-gcc/lib
#MPICPPLINK = -L/openbayes/home/mpich-install/lib/

METISFLAGS = /openbayes/home/PanguLU-main-bak/metis/build/Linux-x86_64/libmetis/libmetis.a

gpu=1
