/*
  MatMul - Efficient matrix multiplication.

  Copyright (c) 2015 by its authors.

  This code is distributed under the BSD3 license. See AUTHORS, LICENSE.
*/

#include <pthread.h>
#include "matmul.h"

/*
  The following globals are used by the pthread slave function.
*/
struct complex ** _A;           // Copy of pointer to matrix A
struct complex ** _B;           // Copy of pointer to matrix B
struct complex ** _C;           // Copy of pointer to matrix C
int _a_dim1, _a_dim2, _b_dim2;  // Copy of dimensions of A and B

/*
  Pthread slave function. Each one carries out a number of dot product
  calculations in the given intervals of rows of A and columns of B.
*/
void * dotProd(void * args) {

  // Slave thread arguments
  struct thread_args * arg = (struct thread_args *) args;

  // Each thread iterates over min(i1, a_dim1) rows of A and
  // min(j1, b_dim2) columns of B
  const int i_lo = arg->i0;
  const int j_lo = arg->j0;
  const int i_hi = (arg->i1 < _a_dim1) ? arg->i1 : _a_dim1;
  const int j_hi = (arg->j1 < _b_dim2) ? arg->j1 : _b_dim2;

  struct complex sum;
  struct complex a, b;
  float a_real, a_imag, b_real, b_imag;

  /*
    The following improves cache hits.
  */
  // Copy row in B into a local array to improve cache performance
  struct complex BI[_a_dim2];

  for (int j = j_lo; (j < j_hi); j++) {

    // Create local B row
    for (int k = 0; (k < _a_dim2); k++) {
      BI[k] = _B[k][j];
    }

    // Perform multiplication
    for (int i = i_lo; (i < i_hi); i++) {
      sum = (struct complex){0.0, 0.0};
      for (int k = 0; (k < _a_dim2); k++) {
        a = _A[i][k];
        b = BI[k];
        a_real = a.real; a_imag = a.imag;
        b_real = b.real; b_imag = b.imag;
        sum.real += (a_real * b_real) - (a_imag * b_imag);
        sum.imag += (a_real * b_imag) + (a_imag * b_real);
      }
      _C[i][j] = sum;
    }
  }

  pthread_exit(NULL);
}

/*
  (Ostensibly) Efficient matrix multiplication routine.
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C,
            int a_dim1, int a_dim2, int b_dim2) {

  // Fall back to matmul() on small input
  if ((a_dim1 < NCORES) && (b_dim2 < NCORES)) {

    struct complex sum;
    struct complex a, b;
    float a_real, a_imag, b_real, b_imag;

    for (int i = 0; (i < a_dim1); i++) {
      for(int j = 0; (j < b_dim2); j++) {
        sum = (struct complex){0.0, 0.0};
        for (int k = 0; (k < a_dim2); k++) {
          // The following code does: sum += A[i][k] * B[k][j];
          a = A[i][k];
          b = B[k][j];
          a_real = a.real; a_imag = a.imag;
          b_real = b.real; b_imag = b.imag;
          sum.real += (a_real * b_real) - (a_imag * b_imag);
          sum.imag += (a_real * b_imag) + (a_imag * b_real);
        }
        C[i][j] = sum;
      }
    }
    return;

  }

  /*
    We now know that one of the matrix dimensions is larger than the number of
    cores we have, so there is (usually) a reasonably large workload to
    distribute amongst all cores (particularly on stoker).
  */

  // Make global copies of parameters for slave threads
  _A = A; _B = B; _C = C;
  _a_dim1 = a_dim1; _a_dim2 = a_dim2; _b_dim2 = b_dim2;

  // Arrays of pthreads and their arguments
  pthread_t threads[NCORES];
  struct thread_args args[NCORES];

  // Distribute largest dimension (either rows or columns) amongst cores
  if (a_dim1 < b_dim2) {

    // Difference in B column indices between threads
    const int DELTA = (b_dim2 + NCORES - 1) / NCORES;

    int col = 0;
    for (int i = 0; (i < NCORES); i++) {
      args[i] = (struct thread_args){0, a_dim1, col, col + DELTA};
      pthread_create(&threads[i], NULL, dotProd, (void *) &args[i]);
      col += DELTA;
    }

  } else {

    // Difference in A row indices between threads
    const int DELTA = (a_dim1 + NCORES - 1) / NCORES;

    int row = 0;
    for (int i = 0; (i < NCORES); i++) {
      args[i] = (struct thread_args){row, row + DELTA, 0, b_dim2};
      pthread_create(&threads[i], NULL, dotProd, (void *) &args[i]);
      row += DELTA;
    }

  }

  // Round up the slaves :(
  for (int i = 0; (i < NCORES); i++) {
    pthread_join(threads[i], NULL);
  }
}
