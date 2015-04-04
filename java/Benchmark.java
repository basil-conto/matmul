/**
 * Benchmarking program for the {@code Matrix.java} matrix library.
 *
 * @author  Basil L. Contovounesios <contovob@tcd.ie>
 */
public class Benchmark {

  public static void main(String[] args) throws IllegalArgumentException {

    Complex[][] A, B, C, ctrl_matrix;
    final int AROWS, ACOLS, BROWS, BCOLS;
    final long time0, time1, time2, ctrl_time, mult_time;
    final double speedup;
    boolean debug = false;
    int argIndex = 0;

    if (args.length == 5) {
      if (!(debug = args[0].equalsIgnoreCase("-d"))) {
        System.err.println("Unrecognised flag: " + args[0]);
        System.exit(1);
      }
      argIndex++;
    } else if (args.length != 4) {
      System.err.println(
        "Usage: java Benchmark <A nrows> <A ncols> <B nrows> <B ncols>");
      System.exit(2);
    }

    AROWS = Integer.parseInt(args[argIndex++]);
    ACOLS = Integer.parseInt(args[argIndex++]);
    BROWS = Integer.parseInt(args[argIndex++]);
    BCOLS = Integer.parseInt(args[argIndex]);

    A = Matrix.random(AROWS, ACOLS);
    B = Matrix.random(BROWS, BCOLS);

    if (debug) {
      System.out.println("A:");
      Matrix.write(A);
      System.out.println("B:");
      Matrix.write(B);
    }

    time0 = System.currentTimeMillis();
    ctrl_matrix = Matrix.mul(A, B);
    time1 = System.currentTimeMillis();
    C = Matrix.fastmul(A, B);
    time2 = System.currentTimeMillis();

    ctrl_time = time1 - time0;
    mult_time = time2 - time1;
    speedup = (float)ctrl_time / mult_time;

    System.out.printf("Control time: %d ms%n", ctrl_time);
    System.out.printf("Fastmul time: %d ms%n", mult_time);

    if ((mult_time > 0) && (ctrl_time > 0)) {
      System.out.printf("Speedup:      %.2fx%n", speedup);
    }

    if (!Matrix.equal(C, ctrl_matrix)) {
      System.err.println("WARNING: Matrices are not equal!");
    }

    if (debug) {
      System.out.println("Control:");
      Matrix.write(ctrl_matrix);
      System.out.println("Fasmul:");
      Matrix.write(C);
    }

  }

}
