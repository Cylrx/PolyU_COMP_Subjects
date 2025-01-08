package hk.edu.polyu.comp.comp2021.assignment1.complex;

public class Complex {

    // Task 5 : add the missing fields

    Rational real, imag;

    // Task 6: Complete the constructor as well as the methods add, subtract, multiply, divide, and toString.
    public Complex(Rational real, Rational imag) {
        this.real = real;
        this.imag = imag;
        this.simplify();
    }

    public Complex add(Complex other) {
        return new Complex(real.add(other.real), imag.add(other.imag));
    }

    public Complex subtract(Complex other) {
        return new Complex(real.subtract(other.real), imag.subtract(other.imag));
    }

    public Complex multiply(Complex other) {
        Rational r = real.multiply(other.real).subtract( imag.multiply(other.imag) );
        Rational i = real.multiply(other.imag).add( other.real.multiply(imag) );
        return new Complex(r, i);
    }

    public Complex divide(Complex other) {
        // you may assume 'other' is never equal to '0+/-0i'.
        Rational a1 = real.multiply(other.real).add( imag.multiply(other.imag) ); // ac + bd
        Rational a2 = imag.multiply(other.real).subtract( real.multiply(other.imag) ); //cb - ad
        Rational b = other.real.multiply(other.real).add (other.imag.multiply(other.imag));
        return new Complex(a1.divide(b), a2.divide(b));
    }

    public void simplify() {
        // Todo: complete the method
        real.simplify();
        imag.simplify();
    }

    public String toString() {
        return "(" + real.toString() + "," + imag.toString() + ")";
    }

    // =========================== Do not change the methods below


    private Rational getReal() {
        return real;
    }

    private void setReal(Rational real) {
        this.real = real;
    }

    private Rational getImag() {
        return imag;
    }

    private void setImag(Rational imag) {
        this.imag = imag;
    }
}
