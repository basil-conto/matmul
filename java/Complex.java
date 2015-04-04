/**
 * Complex number class supporting basic arithmetic operations.
 *
 * @author  Basil L. Contovounesios <contovob@tcd.ie>
 */
public class Complex {

  public double real;
  public double imag;

  public Complex(double re, double im) {
    real = re;
    imag = im;
  }

  public void add(Complex b) {
    real += b.real;
    imag += b.imag;
  }

  public static Complex add(Complex a, Complex b) {
    return new Complex(a.real + b.real, a.imag + b.imag);
  }

  public void mul(Complex b) {
    double re = (real * b.real) - (imag * b.imag);
    double im = (real * b.imag) + (imag * b.real);
    real = re;
    imag = im;
  }

  public static Complex mul(Complex a, Complex b) {
    return new Complex((a.real * b.real) - (a.imag * b.imag),
                       (a.real * b.imag) + (a.imag * b.real));
  }

  @Override
  public String toString() {
    return String.format("(%.2f + %.2fi)", real, imag);
  }

}
