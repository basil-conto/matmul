/*
  Test and timing harness program for developing a dense matrix multiplication
  routine for the CS3014 module using Open MPI.

  All non-MPI code was written by the lecturer of the module as is, as such,
  owned by the university, Trinity College Dublin. It is included purely for the
  purpose of showing the MPI code in action, as was intended for the module.

  The owner of this project has slightly modified the harness and added the MPI
  routine.

  The original version can be found at:
  https://www.scss.tcd.ie/David.Gregg/cs3014/labs/complex-matmul-harness.c

  MPI author:   Basil L. Contovounesios <contovob@tcd.ie>
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <mpi.h>

/*
  Complex unit stored in matrices.
*/
struct complex {
  float real;
  float imag;
};

/*
  The following two definitions of DEBUGGING control whether or not debugging
  information is written out. Defining DEBUG, e.g. via preprocessor options
  at compilation, puts the program into debugging mode. Definining DEBUGGING
  allows for constructs of the form DEBUGGING({ <code> }).
*/
#ifdef DEBUG
# define DEBUGGING(_x) _x
#else
# define DEBUGGING(_x)
#endif

int _id;    // Current process rank
int _np;    // Size of MPI_COMM_WORLD

static const int MASTER      =   0;     // Rank of master node
static const int DEFAULT_TAG =   0;     // Default MPI tag
static const int THRESHOLD   = 128;     // Minimum size for parallelisation

#define IS_MASTER ((_id) == (MASTER))   // Check for master slave

/*
  Create a new empty matrix.
*/
struct complex ** new_empty_matrix(int rows, int cols) {

  struct complex ** result = malloc(sizeof(struct complex *) * rows);
  struct complex * new_matrix = malloc(sizeof(struct complex) * rows * cols);

  for (int i = 0; (i < rows); i++) {
    result[i] = &new_matrix[i * cols];
  }

  return result;
}

/*
  Create a matrix and fill it with random numbers.
*/
struct complex ** gen_random_matrix(int dim1, int dim2) {

  const int RANDOM_RANGE = 512; // Constant power of 2
  struct complex ** result;
  struct timeval seedtime;
  int seed;

  result = new_empty_matrix(dim1, dim2);

  // Use the microsecond part of the current time as a pseudo-random seed 
  gettimeofday(&seedtime, NULL);
  seed = seedtime.tv_usec;
  srandom(seed);

  // Fill the matrix with random numbers
  for (int i = 0; (i < dim1); i++) {
    for (int j = 0; (j < dim2); j++) {
      // Evenly generate values in the range [0, random_range - 1)
      result[i][j].real = (float)(random() % RANDOM_RANGE);
      result[i][j].imag = (float)(random() % RANDOM_RANGE);

      // At no loss of precision, negate the values sometimes so the range is
      // now (-(random_range - 1), random_range - 1)
      if (random() & 1) result[i][j].real = -result[i][j].real;
      if (random() & 1) result[i][j].imag = -result[i][j].imag;
    }
  }

  return result;
}

/*
  Free matrix.
*/
void free_matrix(struct complex ** matrix) {
  free (matrix[0]); // Free the contents
  free (matrix);    // Free the header
}

/*
  Write matrix to stdout.
*/
void write_out(struct complex ** a, int dim1, int dim2) {
  for (int i = 0; (i < dim1); i++) {
    for (int j = 0; (j < dim2 - 1); j++) {
      printf("% .3f + % .3fi ", a[i][j].real, a[i][j].imag);
    }
    printf("% .3f + % .3fi\n", a[i][dim2 - 1].real, a[i][dim2 - 1].imag);
  }
}

/*
  Check the sum of absolute differences is within reasonable epsilon.
*/
void check_result(struct complex ** result, struct complex ** control,
                  int dim1, int dim2) {

  double diff = 0.0;
  double sum_abs_diff = 0.0;
  const double EPSILON = 0.0625;

  for (int i = 0; (i < dim1); i++) {
    for (int j = 0; (j < dim2); j++) {
      diff = abs(control[i][j].real - result[i][j].real);
      sum_abs_diff += diff;
      diff = abs(control[i][j].imag - result[i][j].imag);
      sum_abs_diff += diff;
    }
  }

  if (sum_abs_diff > EPSILON) {
    fprintf(stderr, "WARNING: sum of absolute differences (%f) > EPSILON (%f)\n",
            sum_abs_diff, EPSILON);
  }
}

/*
  Multiply matrix A times matrix B and put result in matrix C.
*/
void ctrl_matmul(struct complex ** A, struct complex ** B, struct complex ** C, 
                 int arows, int acols, int bcols) {

  for (int i = 0; (i < arows); i++) {
    for(int j = 0; (j < bcols); j++) {
      struct complex sum = {0.0, 0.0};
      for (int k = 0; (k < acols); k++) {
        // The following code does: sum += A[i][k] * B[k][j];
        struct complex prod;
        prod.real = A[i][k].real * B[k][j].real - A[i][k].imag * B[k][j].imag;
        prod.imag = A[i][k].real * B[k][j].imag + A[i][k].imag * B[k][j].real;
        sum.real += prod.real;
        sum.imag += prod.imag;
      }
      C[i][j] = sum;
    }
  }
}

/*
  (Ostensibly) Efficient matrix multiplication routine using Open MPI.
*/
void matmul(struct complex ** A, struct complex ** B, struct complex ** C,
            int arows, int acols, int bcols) {

  // Only parallelise large enough matrices.
  if ((arows < THRESHOLD) && (bcols < THRESHOLD)) {

    // Master node carries out sequential multiplication
    if (IS_MASTER) {

      struct complex a, b, sum,             // Local variables for dot products
                     bcol[acols];           // Local transpose of a B col
      float a_real, a_imag, b_real, b_imag; // Local variables for dot products

      // For every B col
      for (int j = 0; (j < bcols); j++) {

        // Make a local (transpose) copy
        for (int k = 0; (k < acols); k++) {
          bcol[k] = B[k][j];
        }

        // Perform dot product making local copies of matrix items in memory
        for (int i = 0; (i < arows); i++) {
          sum = (struct complex){0.0, 0.0};
          for (int k = 0; (k < acols); k++) {
            a = A[i][k];
            b = bcol[k];
            a_real = a.real; a_imag = a.imag;
            b_real = b.real; b_imag = b.imag;
            sum.real += (a_real * b_real) - (a_imag * b_imag);
            sum.imag += (a_real * b_imag) + (a_imag * b_real);
          }
          C[i][j] = sum;
        }
      }
    }

    return;
  }

  // Derived datatypes
  MPI_Datatype mpi_complex, mpi_arow, mpi_crow;

  MPI_Type_contiguous(2, MPI_FLOAT, &mpi_complex);      // struct complex
  MPI_Type_commit(&mpi_complex);

  MPI_Type_contiguous(acols, mpi_complex, &mpi_arow);   // A row
  MPI_Type_commit(&mpi_arow);

  MPI_Type_contiguous(bcols, mpi_complex, &mpi_crow);   // C row
  MPI_Type_commit(&mpi_crow);

  struct complex a, b, sum,             // Local variables for dot products
                 mpi_bcol[acols],       // Array for message passing of B cols
                 mpi_ccol[arows];       // Array for message passing of C cols

  float a_real, a_imag, b_real, b_imag; // Local variables for dot products

  /*
    Distribute A rows or B cols amongst processes. This is done by dividing
    the larger of the two dimensions by the number of processes, NP, we have.
    Due to the properties of integer division, this will always be the minimum
    possible stepping, or delta, of the dimension to send to a process. In cases
    where the larger dimension is not a multiple of NP, however, it is possible,
    at least for a subset of the dimension, to have a delta of one more than the
    previously calculated minimum. The size of this subset is equal to the size
    of the dimension modulo NP.
  */

  if (arows < bcols) {

    const int SMALL_DELTA  = bcols / _np;       // Minimum stepping
    const int LARGE_DELTA  = SMALL_DELTA + 1;   // Larger stepping
    const int LARGE_DELTAS = bcols % _np;       // Number of larger steps

    // Total number of cols due to large stepping
    const int LARGE_COLS   = LARGE_DELTA * LARGE_DELTAS;

    if (IS_MASTER) {

      // Save last stepping for self
      const int LAST_DELTA = bcols - SMALL_DELTA;

      /*
        Broadcast A
      */
      for (int i = 0; (i < arows); i++) {
        MPI_Bcast(A[i], 1, mpi_arow, MASTER, MPI_COMM_WORLD);
      }

      /*
        Scatter B in column-major order
      */
      int col = 0, proc = 1;

      // For each large stepping
      for ( ; (col < LARGE_COLS); col += LARGE_DELTA) {
        // For each column within a large stepping
        for (int delta = 0; (delta < LARGE_DELTA); delta++) {
          // Create and send buffer for B col
          for (int i = 0; (i < acols); i++) {
            mpi_bcol[i] = B[i][col + delta];
          }
          MPI_Send(mpi_bcol, acols, mpi_complex, proc,
                   DEFAULT_TAG, MPI_COMM_WORLD);
        }
        proc++;
      }

      // Do the same for each small stepping
      for ( ; (col < LAST_DELTA); col += SMALL_DELTA) {
        for (int delta = 0; (delta < SMALL_DELTA); delta++) {
          for (int i = 0; (i < acols); i++) {
            mpi_bcol[i] = B[i][col + delta];
          }
          MPI_Send(mpi_bcol, acols, mpi_complex, proc,
                   DEFAULT_TAG, MPI_COMM_WORLD);
        }
        proc++;
      }

      /*
        Perform calculation of one small stepping
      */

      struct complex bcol[acols];   // Local transpose of a B col

      for (int j = LAST_DELTA; (j < bcols); j++) {
        for (int k = 0; (k < acols); k++) {
          bcol[k] = B[k][j];
        }
        for (int i = 0; (i < arows); i++) {
          sum = (struct complex){0.0, 0.0};
          for (int k = 0; (k < acols); k++) {
            a = A[i][k];
            b = bcol[k];
            a_real = a.real; a_imag = a.imag;
            b_real = b.real; b_imag = b.imag;
            sum.real += (a_real * b_real) - (a_imag * b_imag);
            sum.imag += (a_real * b_imag) + (a_imag * b_real);
          }
          C[i][j] = sum;
        }
      }

      /*
        Gather C
      */
      col = 0, proc = 1;
      for ( ; (col < LARGE_COLS); col += LARGE_DELTA) {
        for (int delta = 0; (delta < LARGE_DELTA); delta++) {
          MPI_Recv(mpi_ccol, arows, mpi_complex, proc,
                   DEFAULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (int i = 0; (i < arows); i++) {
            C[i][col + delta] = mpi_ccol[i];
          }
        }
        proc++;
      }
      for ( ; (col < LAST_DELTA); col += SMALL_DELTA) {
        for (int delta = 0; (delta < SMALL_DELTA); delta++) {
          MPI_Recv(mpi_ccol, arows, mpi_complex, proc,
                   DEFAULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for (int i = 0; (i < arows); i++) {
            C[i][col + delta] = mpi_ccol[i];
          }
        }
        proc++;
      }

    } else {    // Slave with arows < bcols

      A = new_empty_matrix(arows, acols);

      // Receive A broadcast
      for (int i = 0; (i < arows); i++) {
        MPI_Bcast(A[i], 1, mpi_arow, MASTER, MPI_COMM_WORLD);
      }

      const int DELTA = (_id <= LARGE_DELTAS) ? LARGE_DELTA : SMALL_DELTA;

      // Allocate space for subsets of B and C
      B = new_empty_matrix(DELTA, acols);
      C = new_empty_matrix(arows, DELTA);

      // Receive B
      for (int i = 0; (i < DELTA); i++) {
        MPI_Recv(B[i], acols, mpi_complex, MASTER,
                 DEFAULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      // Multiply
      for (int i = 0; (i < arows); i++) {
        for (int j = 0; (j < DELTA); j++) {
          sum = (struct complex){0.0, 0.0};
          for (int k = 0; (k < acols); k++) {
            a = A[i][k];
            b = B[j][k];
            a_real = a.real; a_imag = a.imag;
            b_real = b.real; b_imag = b.imag;
            sum.real += (a_real * b_real) - (a_imag * b_imag);
            sum.imag += (a_real * b_imag) + (a_imag * b_real);
          }
          C[i][j] = sum;
        }
      }

      // Send C in column-major order
      for (int j = 0; (j < DELTA); j++) {
        for (int i = 0; (i < arows); i++) {
          mpi_ccol[i] = C[i][j];
        }
        MPI_Send(mpi_ccol, arows, mpi_complex, MASTER,
                 DEFAULT_TAG, MPI_COMM_WORLD);
      }

    }

  } else {  // arows >= bcols

    const int SMALL_DELTA  = arows / _np;       // Minimum stepping
    const int LARGE_DELTA  = SMALL_DELTA + 1;   // Larger stepping
    const int LARGE_DELTAS = arows % _np;       // Number of larger steps

    // Total number of rows due to large stepping
    const int LARGE_ROWS   = LARGE_DELTA * LARGE_DELTAS;

    if (IS_MASTER) {

      // Save last stepping for self
      const int LAST_DELTA = arows - SMALL_DELTA;

      /*
        Broadcast B in column-major order
      */
      for (int j = 0; (j < bcols); j++) {
        for (int i = 0; (i < acols); i++) {
          mpi_bcol[i] = B[i][j];
        }
        MPI_Bcast(mpi_bcol, 1, mpi_arow, MASTER, MPI_COMM_WORLD);
      }

      /*
        Scatter A
      */
      int row = 0, proc = 1;
      for ( ; (row < LARGE_ROWS); row += LARGE_DELTA) {
        for (int delta = 0; (delta < LARGE_DELTA); delta++) {
          MPI_Send(A[row + delta], 1, mpi_arow, proc,
                   DEFAULT_TAG, MPI_COMM_WORLD);
        }
        proc++;
      }
      for ( ; (row < LAST_DELTA); row += SMALL_DELTA) {
        for (int delta = 0; (delta < SMALL_DELTA); delta++) {
          MPI_Send(A[row + delta], 1, mpi_arow, proc,
                   DEFAULT_TAG, MPI_COMM_WORLD);
        }
        proc++;
      }

      /*
        Perform calculation of one small stepping
      */

      struct complex bcol[acols];   // Local transpose of a B col

      for (int j = 0; (j < bcols); j++) {
        for (int k = 0; (k < acols); k++) {
          bcol[k] = B[k][j];
        }
        for (int i = LAST_DELTA; (i < arows); i++) {
          sum = (struct complex){0.0, 0.0};
          for (int k = 0; (k < acols); k++) {
            a = A[i][k];
            b = bcol[k];
            a_real = a.real; a_imag = a.imag;
            b_real = b.real; b_imag = b.imag;
            sum.real += (a_real * b_real) - (a_imag * b_imag);
            sum.imag += (a_real * b_imag) + (a_imag * b_real);
          }
          C[i][j] = sum;
        }
      }

      // Gather C
      row = 0, proc = 1;
      for ( ; (row < LARGE_ROWS); row += LARGE_DELTA) {
        for (int delta = 0; (delta < LARGE_DELTA); delta++) {
          MPI_Recv(C[row + delta], 1, mpi_crow, proc,
                   DEFAULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        proc++;
      }
      for ( ; (row < LAST_DELTA); row += SMALL_DELTA) {
        for (int delta = 0; (delta < SMALL_DELTA); delta++) {
          MPI_Recv(C[row + delta], 1, mpi_crow, proc,
                   DEFAULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        proc++;
      }

    } else {    // Slave with arows >= bcols

      B = new_empty_matrix(bcols, acols);   // Tranpose B

      // Receive B broadcast
      for (int i = 0; (i < bcols); i++) {
        MPI_Bcast(B[i], 1, mpi_arow, MASTER, MPI_COMM_WORLD);
      }

      const int DELTA = (_id <= LARGE_DELTAS) ? LARGE_DELTA : SMALL_DELTA;

      // Allocate space for subsets of B and C
      A = new_empty_matrix(DELTA, acols);
      C = new_empty_matrix(DELTA, bcols);

      // Receive A
      for (int i = 0; (i < DELTA); i++) {
        MPI_Recv(A[i], 1, mpi_arow, MASTER,
                 DEFAULT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }

      // Multiply
      for (int i = 0; (i < DELTA); i++) {
        for (int j = 0; (j < bcols); j++) {
          sum = (struct complex){0.0, 0.0};
          for (int k = 0; (k < acols); k++) {
            a = A[i][k];
            b = B[j][k];
            a_real = a.real; a_imag = a.imag;
            b_real = b.real; b_imag = b.imag;
            sum.real += (a_real * b_real) - (a_imag * b_imag);
            sum.imag += (a_real * b_imag) + (a_imag * b_real);
          }
          C[i][j] = sum;
        }
      }

      // Send C in row-major order
      for (int i = 0; (i < DELTA); i++) {
        MPI_Send(C[i], 1, mpi_crow, MASTER, DEFAULT_TAG, MPI_COMM_WORLD);
      }

    }   // Slave with arows <= bcols

  }     // arows <= bcols

  // Be free, datatypes!
  MPI_Type_free(&mpi_arow);
  MPI_Type_free(&mpi_crow);
  MPI_Type_free(&mpi_complex);
}

/*
  Returns the difference, in microseconds, between the two given times.
*/
long long time_diff(struct timeval * start, struct timeval *end) {
  return ((end->tv_sec - start->tv_sec) * 1000000L) +
         (end->tv_usec - start->tv_usec);
}

/*
  Main harness.
*/
int main(int argc, char ** argv) {

  struct complex ** A, ** B, ** C;
  struct complex ** ctrl_matrix;
  int arows, acols, brows, bcols;

  struct timeval time0, time1, time2;
  long long ctrl_time, mult_time;
  double speedup;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &_id);
  MPI_Comm_size(MPI_COMM_WORLD, &_np);

  /*
    Master node carries out error handling and generates matrices
  */

  if (IS_MASTER) {
    if (argc != 5) {
      fputs("Usage: matMul <A nrows> <A ncols> <B nrows> <B ncols>\n", stderr);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  /*
    All nodes parse their own matrix size arguments
  */

  arows = atoi(argv[1]);
  acols = atoi(argv[2]);
  brows = atoi(argv[3]);
  bcols = atoi(argv[4]);

  if (IS_MASTER) {

    // Check the matrix sizes are compatible
    if (acols != brows) {
      fprintf(stderr, "FATAL number of columns of A (%d) does not "
              "match number of rows of B (%d)\n", acols, brows);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate the matrices
    A = gen_random_matrix(arows, acols);
    B = gen_random_matrix(brows, bcols);
    C =  new_empty_matrix(arows, bcols);
    ctrl_matrix = new_empty_matrix(arows, bcols);

    DEBUGGING({
      puts("Matrix A:");
      write_out(A, arows, acols);
      puts("\nMatrix B:");
      write_out(B, brows, bcols);
      puts("");
    })

    // Record control start time
    gettimeofday(&time0, NULL);

    ctrl_matmul(A, B, ctrl_matrix, arows, acols, bcols);

    // Record matmul start time
    gettimeofday(&time1, NULL);

  }

  matmul(A, B, C, arows, acols, bcols);

  if (IS_MASTER) {

    // Record finishing time
    gettimeofday(&time2, NULL);

    // Compute elapsed times and speedup factor
    ctrl_time = time_diff(&time0, &time1);
    mult_time = time_diff(&time1, &time2);
    speedup   = (float)ctrl_time / mult_time;

    printf("Control time: %lld μs\n", ctrl_time);
    printf("Matmul  time: %lld μs\n", mult_time);

    if ((mult_time > 0) && (ctrl_time > 0)) {
      printf("Speedup:      %.2fx\n", speedup);
    }

    // Now check that team_matmul() gives the same answer as the control
    check_result(C, ctrl_matrix, arows, bcols);

    DEBUGGING({
      puts("\nResultant matrix (control):");
      write_out(ctrl_matrix, arows, bcols);
      puts("\nResultant matrix (matmul):");
      write_out(C, arows, bcols);
    })

    // Free all matrices
    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
    free_matrix(ctrl_matrix);

  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
