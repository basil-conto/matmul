/*
  MatMul - Efficient matrix multiplication.

  Copyright (c) 2015 by its authors.

  This code is distributed under the BSD3 license. See AUTHORS, LICENSE.
*/

#include <pmmintrin.h>  // Optionally smmintrin.h for _mm_extract_ps()
#include <pthread.h>
#include "matmul.h"

/*
  The following globals are used by the pthread slave function.
*/
struct complex ** _A;       // Copy of pointer to matrix A
struct complex ** _B;       // Copy of pointer to matrix B
struct complex ** _C;       // Copy of pointer to matrix C
int _arows, _acols, _bcols; // Copy of dimensions of A and B

/*
  Pthread slave function. Each one carries out a number of dot product
  calculations in the given intervals of rows of A and columns of B.
*/
void * dotProd(void * args) {

  // Slave thread arguments
  struct thread_args * arg = (struct thread_args *) args;

  // Each thread iterates over min(i1, arows) rows of A and
  // min(j1, bcols) columns of B
  const int i_lo = arg->i0;
  const int j_lo = arg->j0;
  const int i_hi = (arg->i1 < _arows) ? arg->i1 : _arows;
  const int j_hi = (arg->j1 < _bcols) ? arg->j1 : _bcols;
  const int k_hi = _acols - 4;

  struct complex a, b, tmp, sum;
  float a_real, a_imag, b_real, b_imag;

  __m128 a0, a1, b0, b1, a0xb0, a1xb1, b0xa0, b1xa1, sub, add, res;

  // Copy column in B into a local array to improve cache performance
  struct complex bcol[_acols] __attribute__ ((aligned (16)));

  for (int j = j_lo; (j < j_hi); j++) {

    // Create local B row
    for (int k = 0; (k < _acols); k++) {
      bcol[k] = _B[k][j];
    }

    // Perform multiplication
    for (int i = i_lo; (i < i_hi); i++) {

      sum = (struct complex){0.0, 0.0};

      int k = 0;
      for ( ; (k <= k_hi); k += 4) {

        a0 = _mm_loadu_ps((float *) &_A[i][k]);     // [ar0, ai0, ar1, ai1]
        a1 = _mm_loadu_ps((float *) &_A[i][k + 2]); // [ar2, ai2, ar3, ai3]

        b0 = _mm_load_ps((float *) &bcol[k]);       // [br0, bi0, br1, bi1]
        b1 = _mm_load_ps((float *) &bcol[k + 2]);   // [br2, bi2, br3, bi3]

        a0xb0 = _mm_mul_ps(a0, b0); // [ar0(br0), ai0(bi0), ar1(br1), ai1(bi1)]
        a1xb1 = _mm_mul_ps(a1, b1); // [ar2(br2), ai2(bi2), ar3(br3), ai3(bi3)]

        // Swap lower and upper real-imag pairs
        b0 = _mm_shuffle_ps(b0, b0, _MM_SHUFFLE(2, 3, 0, 1));
        b1 = _mm_shuffle_ps(b1, b1, _MM_SHUFFLE(2, 3, 0, 1));

        b0xa0 = _mm_mul_ps(a0, b0); // [ar0(bi0), ai0(br0), ar1(bi1), ai1(br1)]
        b1xa1 = _mm_mul_ps(a1, b1); // [ar2(bi2), ai2(br2), ar3(bi3), ai3(br3)]

        sub = _mm_hsub_ps(a0xb0, a1xb1);            // [cr0, cr1, cr2, cr3]
        add = _mm_hadd_ps(b0xa0, b1xa1);            // [ci0, ci1, ci2, ci3]

        res = _mm_hadd_ps(sub, add);    // [cr0+cr1, cr2+cr3, ci0+ci1, ci2+ci3]
        res = _mm_hadd_ps(res, res);    // [cr0+cr1+cr2+cr3, ci0+ci1+ci2+ci3,
                                        //  cr0+cr1+cr2+cr3, ci0+ci1+ci2+ci3]

        /*
          Choose between:

            a) storing lower two elements in a temporary struct complex
        */

        _mm_storel_pi((__m64 *) &tmp, res);
        sum.real += tmp.real;
        sum.imag += tmp.imag;

        /*
            b) individually extracting lower two elements
        */

        // sum.real += (float)_mm_extract_ps(res, 0x0);
        // sum.imag += (float)_mm_extract_ps(res, 0x1);
      }

      for ( ; (k < _acols); k++) {
        a = _A[i][k];
        b =  bcol[k];
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
            int arows, int acols, int bcols) {

  // Fall back to plain multiplication on small input
  if ((arows < NCORES) && (bcols < NCORES)) {

    struct complex sum;
    struct complex a, b;
    float a_real, a_imag, b_real, b_imag;

    for (int i = 0; (i < arows); i++) {
      for(int j = 0; (j < bcols); j++) {
        sum = (struct complex){0.0, 0.0};
        for (int k = 0; (k < acols); k++) {
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
  _arows = arows; _acols = acols; _bcols = bcols;

  // Arrays of pthreads and their arguments
  pthread_t threads[NCORES];
  struct thread_args args[NCORES];

  // Distribute largest dimension (either rows or columns) amongst cores
  if (arows < bcols) {

    /*
      Choose between having:

        a) two different steppings to guarantee all threads are busy at the cost
           of more complex sequential code
    */

    const int SMALL_DELTA  = bcols / NCORES;    // Minimum stepping
    const int LARGE_DELTA  = SMALL_DELTA + 1;   // Larger stepping
    const int LARGE_DELTAS = bcols % NCORES;    // Number of larger steps

    int i = 0, col = 0;
    for ( ; (i < LARGE_DELTAS); i++) {
      args[i] = (struct thread_args){0, arows, col, col + LARGE_DELTA};
      pthread_create(&threads[i], NULL, dotProd, (void *) &args[i]);
      col += LARGE_DELTA;
    }
    for ( ; (i < NCORES); i++) {
      args[i] = (struct thread_args){0, arows, col, col + SMALL_DELTA};
      pthread_create(&threads[i], NULL, dotProd, (void *) &args[i]);
      col += SMALL_DELTA;
    }

    /*
        b) a single stepping which sometimes results in idle threads but less
           complex sequential code
    */

    // // Difference in B column indices between threads
    // const int DELTA = (bcols + NCORES - 1) / NCORES;

    // int col = 0;
    // for (int i = 0; (i < NCORES); i++) {
    //   args[i] = (struct thread_args){0, arows, col, col + DELTA};
    //   pthread_create(&threads[i], NULL, dotProd, (void *) &args[i]);
    //   col += DELTA;
    // }

  } else {

    /*
      Choose between having:

        a) two different steppings to guarantee all threads are busy at the cost
           of more complex sequential code
    */

    const int SMALL_DELTA  = arows / NCORES;    // Minimum stepping
    const int LARGE_DELTA  = SMALL_DELTA + 1;   // Larger stepping
    const int LARGE_DELTAS = arows % NCORES;    // Number of larger steps

    int i = 0, row = 0;
    for ( ; (i < LARGE_DELTAS); i++) {
      args[i] = (struct thread_args){row, row + LARGE_DELTA, 0, bcols};
      pthread_create(&threads[i], NULL, dotProd, (void *) &args[i]);
      row += LARGE_DELTA;
    }
    for ( ; (i < NCORES); i++) {
      args[i] = (struct thread_args){row, row + SMALL_DELTA, 0, bcols};
      pthread_create(&threads[i], NULL, dotProd, (void *) &args[i]);
      row += SMALL_DELTA;
    }

    /*
        b) a single stepping which sometimes results in idle threads but less
           complex sequential code
    */

    // // Difference in A row indices between threads
    // const int DELTA = (arows + NCORES - 1) / NCORES;

    // int row = 0;
    // for (int i = 0; (i < NCORES); i++) {
    //   args[i] = (struct thread_args){row, row + DELTA, 0, bcols};
    //   pthread_create(&threads[i], NULL, dotProd, (void *) &args[i]);
    //   row += DELTA;
    // }

  }

  // Round up the slaves :(
  for (int i = 0; (i < NCORES); i++) {
    pthread_join(threads[i], NULL);
  }
}
