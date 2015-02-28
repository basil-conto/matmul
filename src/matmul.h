
/*
  Header defining macros and datatypes used by the MatMul program, as well as
  the prototype for the core multiplication routine.

  Copyright (c) 2015 by its authors.

  This code is distributed under the BSD3 license. See AUTHORS, LICENSE.
*/

#ifndef MATMUL_H
# define MATMUL_H

# ifndef MATRIX_H
#  include "matrix.h"
# endif

/*
  The following definition reflects the number of online cores on the system
  and determines the maximum number of pthreads created. It defaults to 64,
  which is the number of cores on the target machine stoker. It is intended to
  be defined at compilation via a preprocessor option if run on a different
  target.
*/
# ifndef NCORES
#  define NCORES 64
# endif

/*
  Matrix indices passed to pthread slave function. Each slave takes on the rows
  of A in the interval [i0, i1) and the columns of B in the interval [j0, j1).
*/
struct thread_args {
  int i0; int i1;
  int j0; int j1;
};

/*
  (Ostensibly) Efficient matrix multiplication routine.
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C,
            int a_dim1, int a_dim2, int b_dim2);

#endif // MATMUL_H
