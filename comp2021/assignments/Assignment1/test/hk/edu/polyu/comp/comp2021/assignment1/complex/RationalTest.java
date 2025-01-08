package hk.edu.polyu.comp.comp2021.assignment1.complex;

import org.junit.Test;

import static org.junit.Assert.*;

public class RationalTest {
    @Test
    public void testConstructor_01() {
        Rational r1 = new Rational(1, 2);
        r1.simplify();
        assertEquals("1/2", r1.toString());
    }

    @Test
    public void testSimplify_01() {
        Rational r1 = new Rational(4, 10);
        r1.simplify();
        assertEquals("2/5", r1.toString());
    }

    @Test
    public void testAddition() {
        Rational r1 = new Rational(1, 2);
        Rational r2 = new Rational(1, 3);

        Rational rSUm = r1.add(r2);
        rSUm.simplify();

        assertEquals("5/6", rSUm.toString());
    }

    @Test
    public void testSubstraction() {
        Rational r1 = new Rational(6, 7);
        Rational r2 = new Rational(8, 8);

        Rational rSub = r1.subtract(r2);
        rSub.simplify();

        System.out.println(rSub.toString());
        //assertEquals("5/12", rSub.toString());
    }

    @Test
    public void testMuliplication() {
        Rational r1 = new Rational(7, 8);
        Rational r2 = new Rational(5, 6);

        Rational rMul = r1.multiply(r2);
        rMul.simplify();

        assertEquals("35/48", rMul.toString());
    }

    @Test
    public void testDevision() {
        Rational r1 = new Rational(2, 3);
        Rational r2 = new Rational(3, 4);

        Rational rDiv = r1.divide(r2);
        rDiv.simplify();

        assertEquals("8/9", rDiv.toString());
    }

}
