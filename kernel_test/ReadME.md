# compile

set up the cuda path in make.inc ,then 

`make`

# run

You should make sure that all test matrices are in file test_matrices. The test code requires PanguLU linkage to generate matrix blocks through PanguLU and then test each set of matrices with the test code.

The generated performance data is stored in . /test/csv_out

## run with single test

`cd test`

### SpGEMM

`./ssssm -A A_filename -B B_filename -C C_filename`

### GETRF

`./getrf -A A_filename -L L_filename -U U_filename`


### GESSM

`./gessm -B B_filename -L L_filename`

### TSTRF

`./tstrf -B B_filename -U U_filename`

## run with bash.sh

### SpGEMM

`bash ssssm_run.sh`

### GETRF

`bash getrf_run.sh`


### GESSM

`bash gessm_run.sh`

### TSTRF

`bash tstrf_run.sh`

### All kernels

`bash sum_run.sh`








