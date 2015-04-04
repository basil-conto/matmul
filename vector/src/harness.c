/*
  Test and timing harness program for developing a dense matrix multiplication
  routine for the CS3014 module.

  This program was written by the lecturer of the module and is, as such, owned
  by the university, Trinity College Dublin.

  It has been slightly modified by the authors of the MatMul program
  (see AUTHORS) and is included purely for the purpose of showing the MatMul
  program in action, as was intended for the module.

  The original version can be found at:
  https://www.scss.tcd.ie/David.Gregg/cs3014/labs/complex-matmul-harness.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "complex.h"
#include "matmul.h"

/*
  The following two definitions of DEBUGGING control whether or not debugging
  information is written out. Defining DEBUG, e.g. via preprocessor options
  at compilation, puts the program into debugging mode.
*/
#ifdef DEBUG
# define DEBUGGING(_x) _x
#else
# define DEBUGGING(_x)
#endif

/*
  Write matrix to stdout.
*/
void write_out(struct complex ** a, int dim1, int dim2) {
  for (int i = 0; (i < dim1); i++) {
    for (int j = 0; (j < dim2 - 1); j++) {
      printf("%f + %fi ", a[i][j].real, a[i][j].imag);
    }
    printf("%f + %fi\n", a[i][dim2 - 1].real, a[i][dim2 - 1].imag);
  }
}

/*
  Create a new empty matrix.
*/
struct complex ** new_empty_matrix(int dim1, int dim2) {

  struct complex ** result = malloc(sizeof(struct complex*) * dim1);
  struct complex * new_matrix = malloc(sizeof(struct complex) * dim1 * dim2);

  for (int i = 0; (i < dim1); i++) {
    result[i] = &new_matrix[i * dim2];
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
  Create a matrix and fill it with random numbers.
*/
struct complex ** gen_random_matrix(int dim1, int dim2) {

  const int random_range = 512;
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
      result[i][j].real = (float)(random() % random_range);
      result[i][j].imag = (float)(random() % random_range);

      // At no loss of precision, negate the values sometimes so the range is
      // now (-(random_range - 1), random_range - 1)
      if (random() & 1) result[i][j].real = -result[i][j].real;
      if (random() & 1) result[i][j].imag = -result[i][j].imag;
    }
  }

  return result;
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
void control_matmul(struct complex ** A, struct complex ** B, struct complex ** C, 
                    int a_dim1, int a_dim2, int b_dim2) {

  for (int i = 0; (i < a_dim1); i++) {
    for(int j = 0; (j < b_dim2); j++) {
      struct complex sum = {0.0, 0.0};
      for (int k = 0; (k < a_dim2); k++) {
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
  long long ctrl_time, mult_time;
  int a_dim1, a_dim2, b_dim1, b_dim2;
  struct timeval time0, time1, time2;
  double speedup;

  if (argc != 5) {
    fputs("Usage: matMul <A nrows> <A ncols> <B nrows> <B ncols>\n", stderr);
    return EXIT_FAILURE;
  } else  {
    a_dim1 = atoi(argv[1]);
    a_dim2 = atoi(argv[2]);
    b_dim1 = atoi(argv[3]);
    b_dim2 = atoi(argv[4]);
  }

  // Check the matrix sizes are compatible
  if (a_dim2 != b_dim1) {
    fprintf(stderr, "FATAL number of columns of A (%d) does not "
                    "match number of rows of B (%d)\n", a_dim2, b_dim1);
    return EXIT_FAILURE;
  }

  // Allocate the matrices
  A = gen_random_matrix(a_dim1, a_dim2);
  B = gen_random_matrix(b_dim1, b_dim2);
  C = new_empty_matrix(a_dim1, b_dim2);
  ctrl_matrix = new_empty_matrix(a_dim1, b_dim2);

  DEBUGGING({
    puts("Matrix A:");
    write_out(A, a_dim1, a_dim2);
    puts("\nMatrix B:");
    write_out(B, b_dim1, b_dim2);
    puts("");
  })

  // Record control start time
  gettimeofday(&time0, NULL);

  // Use a simple matmul routine to produce control result
  control_matmul(A, B, ctrl_matrix, a_dim1, a_dim2, b_dim2);

  DEBUGGING({
    puts("Resultant matrix (control):");
    write_out(ctrl_matrix, a_dim1, b_dim2);
    puts("");
  })

  // Record start time
  gettimeofday(&time1, NULL);

  // Perform fast matrix multiplication
  matmul(A, B, C, a_dim1, a_dim2, b_dim2);

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
  check_result(C, ctrl_matrix, a_dim1, b_dim2);

  DEBUGGING({
    puts("Resultant matrix (matmul):");
    write_out(C, a_dim1, b_dim2);
  })

  // Free all matrices
  free_matrix(A);
  free_matrix(B);
  free_matrix(C);
  free_matrix(ctrl_matrix);

  return EXIT_SUCCESS;
}
