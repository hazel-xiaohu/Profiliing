# PanguLU

 **PanguLU** is an open source software package that uses a block sparse structure to solve linear systems A*X=B. It uses a sparse matrix block LU decomposition algorithm. The sparse matrix block LU decomposition algorithm is an algorithm that splits the sparse matrix into multiple sparse matrix blocks, and uses sflu, sparse triangle solution sptrsv, sparse matrix matrix multiplication spgemm and other kernels among the sparse matrix blocks. The algorithm can be used in heterogeneous The distributed platform operates accurately and efficiently.

-------------------

## Introduction

PanguLU is mainly deployed at distributed multi-GPU parallel clusters. PanguLU is implemented in the use of MPI and CUDA,  MPI is used for communication and CUDA is used for GPU computing . We have implemented multi-GPU acceleration and are still actively accelerating.


## Structure of code

```
PanguLU/README      instructions on installation
PanguLU/src         C and CUDA source code, to be compiled into libpangulu.so
PanguLU/test        testing code
PanguLU/icnlude     contains headers archieve libpangulu.a
PanguLU/lib         contains library archieve libpangulu.a
PanguLU/Makefile    top-level Makefile that does installation and testing
PanguLU/make.inc    compiler, compiler flags included in all Makefiles
PanguLU/matrix      a download matrices script and download matrices in here
PanguLU/test_kernel source code and test files for testing sparse kernel perfomance 
PanguLU/script      some python scripts for plotting and shell execution scripts 
PanguLU/figure      Storege of the figures from python scripts for plotting
PanguLU/data        Storage of the data from shell script execution and drawing in csv format 
```

## Installation
we use the method is to use make automatic build system.
You can build new code by running our script file new.sh.

## Reproducibility of Experiments

We have included seven main running scripts in the folder script, which are included below:

```
figure7.sh (3 minutes)
figure7_origin.sh (3 days)
figure11.sh (30 minutes)
figure12.sh (3 hours)
figure12_origin.sh (16 hours)
figure13.sh (30 minutes)
figure14.sh (1 hours and 30 minutes)
```
The rest of the other scripts are responsible for helping the above scripts do their job.

## Caution

about Reproducibility of Experiments：

To simplify the runtime, we tested the code for figure7 ahead of time and saved the results in the kernel_test\test\csv_out folder, where you can reproduce the experiment completely with figure7_origin.sh.

Since process 1,2,4 in the MPI programme is slow, we tested the first 1,2,4 processes in figure12 in advance and saved the data in the data folder, you can reproduce the experiment completely with figure12_origin.sh, this script deletes all the data at the beginning.

about run the MPI code：
You need to change some of the MPI parameters for a specific machine in script/run.sh, in particular add the --hostfile parameter to map the MPI processes to different nodes, be aware that if a node has X A100 Nvidia GPUs then you need to map X processes to that node to do the calculations. For example, if a node has 4 A100 Nvidia GPUs, 4 processes need to be mapped to a single node.

## all.sh

Once you have modified the above caution, we provide an all.sh to run all the programs, and you can check the graphs in the FIGURE after 5 hours and 30 minutes to see the results of our reproduction.

We are happy to help you reproduce our experiments, so if you experience any problems, feel free to contact us. If you encounter a bug that is difficult to resolve, we can help you reproduce the experiment remotely from your machine if it is convenient.

 





