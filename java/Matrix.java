import java.util.Random;

/**
 * Matrix library, particularly focussing on efficient multiplication.
 *
 * @author  Basil L. Contovounesios <contovob@tcd.ie>
 */
public class Matrix {

  public static final double EPSILON    = 0.0625;
  public static final double RAND_RANGE = 512.0;

  private static void checkDimen(int dim0, int dim1)
      throws IllegalArgumentException {
    if ((dim0 < 1) || (dim1 < 1)) {
      throw new IllegalArgumentException(
        "Matrix dimensions must be greater than zero.");
    }
  }

  public static Complex[][] empty(int rows, int cols)
      throws IllegalArgumentException {
    checkDimen(rows, cols);
    Complex[][] result = new Complex[rows][cols];
    for (int i = 0; (i < rows); i++) {
      for (int j = 0; (j < cols); j++) {
        result[i][j] = new Complex(0.0, 0.0);
      }
    }
    return result;
  }

  public static Complex[][] random(int rows, int cols)
      throws IllegalArgumentException {
    checkDimen(rows, cols);
    Random gen = new Random();
    double real, imag;

    Complex[][] result = new Complex[rows][cols];
    for (int i = 0; (i < rows); i++) {
      for (int j = 0; (j < cols); j++) {
        real = gen.nextDouble() * RAND_RANGE;
        imag = gen.nextDouble() * RAND_RANGE;
        if ((gen.nextInt() & 1) == 0) real = -real;
        if ((gen.nextInt() & 1) == 0) imag = -imag;
        result[i][j] = new Complex(real, imag);
      }
    }
    return result;
  }

  public static void write(Complex[][] matrix) {
    for (Complex[] row : matrix) {
      final int LAST_COL = row.length - 1;
      for (int j = 0; (j < LAST_COL); j++) {
        System.out.print(row[j] + " ");
      }
      System.out.println(row[LAST_COL]);
    }
  }

  public static boolean equal(Complex[][] control, Complex[][] compare) {
    double abs_diff = 0.0;
    for (int i = 0; (i < control.length); i++) {
      for (int j = 0; (j < control[i].length); j++) {
        Complex ctrl = control[i][j];
        Complex cmpr = compare[i][j];
        abs_diff += Math.abs(ctrl.real - cmpr.real) +
                    Math.abs(ctrl.imag - cmpr.imag);
      }
    }
    return (abs_diff < EPSILON);
  }

  public static Complex[][] mul(Complex[][] A, Complex[][] B)
      throws IllegalArgumentException{
    final int AROWS = A.length;
    final int ACOLS = B.length;

    checkDimen(AROWS, ACOLS);

    if (A[0].length != ACOLS) {
      throw new IllegalArgumentException(String.format(
          "Columns of A (%d) do not match rows of B (%d)", A[0].length, ACOLS));
    }

    final int BCOLS = B[0].length;

    Complex[][] C = new Complex[AROWS][BCOLS];

    for (int i = 0; (i < AROWS); i++) {
      for (int j = 0; (j < BCOLS); j++) {
        Complex sum = new Complex(0.0, 0.0);
        for (int k = 0; (k < ACOLS); k++) {
          sum.add(Complex.mul(A[i][k], B[k][j]));
        }
        C[i][j] = sum;
      }
    }
    return C;
  }

  public static Complex[][] fastmul(Complex[][] A, Complex[][] B) {
    return (new FastMul(A, B)).fastmul();
  }

  private static class FastMul {

    private final Complex[][] A, B;
    private Complex[][] C;

    private final int AROWS, ACOLS, BCOLS;
    private final int CORES;

    private Thread[] threads;

    private FastMul(Complex[][] a, Complex[][] b) {
      AROWS = a.length;
      ACOLS = b.length;

      if ((AROWS < 1) || (ACOLS < 1)) {
        throw new IllegalArgumentException(
          "Matrix dimensions must be greater than zero.");
      }

      if (a[0].length != ACOLS) {
        throw new IllegalArgumentException(String.format(
          "Columns of A (%d) do not match rows of B (%d)", a[0].length, ACOLS));
      }

      A = a;
      B = b;
      BCOLS = B[0].length;
      CORES = Runtime.getRuntime().availableProcessors();
      threads = new Thread[CORES];
      C = new Complex[AROWS][BCOLS];
    }

    private Complex[][] fastmul() {
      final int DELTA;

      if (AROWS < BCOLS) {
        DELTA = (BCOLS + CORES - 1) / CORES;

        int col = 0;
        for (int i = 0; (i < CORES); i++) {
          (threads[i] = new Thread(
            new FastThread(0, AROWS, col, col += DELTA))).start();
        }

      } else {
        DELTA = (AROWS + CORES - 1) / CORES;

        int row = 0;
        for (int i = 0; (i < CORES); i++) {
          (threads[i] = new Thread(
            new FastThread(row, row += DELTA, 0, BCOLS))).start();
        }
      }

      for (Thread t : threads) {
        try {
          t.join();
        } catch (InterruptedException ie) {
          ie.printStackTrace();
        }
      }

      return C;
    }

    private class FastThread implements Runnable {

      private final int i_lo, i_hi, j_lo, j_hi;
      private Complex[] bcol = new Complex[ACOLS];

      private FastThread(int il, int ih, int jl, int jh) {
        i_lo = il;
        j_lo = jl;
        i_hi = (ih < AROWS) ? ih : AROWS;
        j_hi = (jh < BCOLS) ? jh : BCOLS;
      }

      public void run() {
        double real, imag;
        Complex a, b;

        for (int j = j_lo; (j < j_hi); j++) {
          for (int k = 0; (k < ACOLS); k++) {
            bcol[k] = B[k][j];
          }
          for (int i = i_lo; (i < i_hi); i++) {
            real = imag = 0.0;
            for (int k = 0; (k < ACOLS); k++) {
              a = A[i][k];
              b = bcol[k];
              real += (a.real * b.real) - (a.imag * b.imag);
              imag += (a.real * b.imag) + (a.imag * b.real);
            }
            C[i][j] = new Complex(real, imag);
          }
        }
      } // run()

    } // FastThread

  } // FastMul

} // Matrix
