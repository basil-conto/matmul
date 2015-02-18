# MatMul

Parallel matrix multiplication.

### Introduction

The `run` script invokes `make` with the necessary flags. The `makefile`, in
turn, compiles `matMul.c` with a preprocessor option that defines `NCORES` as
the number of online cores on the system. This determines the maximum number of
pthreads spawned.

### Normal usage

Run with

    ./run <A nrows> <A ncols> <B nrows> <B ncols>

or

    ./run <size>

which multiplies two `size x size` matrices.

### Debugging

Run in debug mode with

    ./run -d <A nrows> <A ncols> <B nrows> <B ncols>

or

    ./run -d <size>

as usual.
