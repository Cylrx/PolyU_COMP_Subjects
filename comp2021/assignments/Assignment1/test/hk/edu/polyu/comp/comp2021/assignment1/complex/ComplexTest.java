package hk.edu.polyu.comp.comp2021.assignment1.complex;

import org.junit.Test;

import static org.junit.Assert.*;

public class ComplexTest {
    @Test
    public void testConstructor() {
        Rational real = new Rational(1, 2);
        Rational imag = new Rational(1, 3);

        Complex c1 = new Complex(real, imag);
        c1.simplify();

        assertEquals("(1/2,1/3)", c1.toString());
    }

    @Test
    public void testAddition() {
        Complex c1 = new Complex(new Rational(1, 2), new Rational(1, 3));
        Complex c2 = new Complex(new Rational(2, 3), new Rational(1, 4));

        Complex cSum = c1.add(c2);
        cSum.simplify();

        assertEquals("(7/6,7/12)", cSum.toString());
    }

    @Test
    public void testSubstraction() {
        Complex c1 = new Complex(new Rational(1, 2), new Rational(2, 3));
        Complex c2 = new Complex(new Rational(1, 3), new Rational(1, 4));

        Complex cSum = c1.subtract(c2);
        cSum.simplify();

        assertEquals("(1/6,5/12)", cSum.toString());
    }

    @Test
    public void testMultiplication() {
        Complex c1 = new Complex(new Rational(1, 4), new Rational(3, 7));
        Complex c2 = new Complex(new Rational(1, 5), new Rational(5, 8));

        Complex cMul = c1.multiply(c2);
        cMul.simplify();

        assertEquals("(-61/280,271/1120)", cMul.toString());
    }

    @Test
    public void testDivision(){
        Complex c1 = new Complex(new Rational(2, 5), new Rational(3, 7));
        Complex c2 = new Complex(new Rational(1, 3), new Rational(1, 6));

        Complex cDiv = c1.divide(c2);
        cDiv.simplify();

        assertEquals("(258/175,96/175)", cDiv.toString());
    }

}
