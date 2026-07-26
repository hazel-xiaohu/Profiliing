#ifndef PANGULU_H
#define PANGULU_H

#include <string.h>
#include <sys/time.h>
#include <sys/types.h>
#include <math.h>
/*
    this function is panguLU test file
    The input parameters are as follows:
    -np is MPI process numbers.
    -NB is the matrix block size.
    -P is number of row nodes.
    -Q is number of col nodes.
    -F is the solution matrix name.
    spicially ,that the solution matrix must be decomposed symbolically, and np must be equal to P times Q .
*/
    void pangulu(int ARGC, char **ARGV);


#endif
