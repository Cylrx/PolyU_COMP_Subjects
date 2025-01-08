package hk.edu.polyu.comp.comp2021.assignment1.complex;

public class Rational {

    // Task 3: add the missing fields

    int a; // numerator
    int b; //denominator

    // Task 4: 	Complete the constructor and
    // the methods add, subtract, multiply, divide, simplify, and toString.

    public Rational(int a, int b){
        // Todo: complete the constructor
        this.a = a;
        this.b = b;

        if (this.b < 0) {
            this.a = -this.a;
            this.b = -this.b;
        }
        this.simplify();
    }

    public Rational add(Rational other){
        int g = gcd(b, other.b);
        int m1 = b / g, m2 = other.b / g;
        Rational res = new Rational(a * m2 + other.a * m1, g * m1 * m2);
        res.simplify();
        return res;
    }

    public Rational subtract(Rational other){
        int g = gcd(b, other.b);
        int m1 = b / g, m2 = other.b / g;
        Rational res = new Rational(a * m2 - other.a * m1, g * m1 * m2);
        res.simplify();
        return res;
    }

    public Rational multiply(Rational other){
        int g1 = gcd(a, other.b);
        int g2 = gcd(b, other.a);
        return new Rational((a / g1) * (other.a / g2), (b / g2) * (other.b / g1));
    }

    public Rational divide(Rational other){
        int g1 = gcd(a, other.a);
        int g2 = gcd(b, other.b);
        return new Rational((a / g1) * (other.b / g2), (b / g2) * (other.a / g1));
    }

    public String toString(){
        return a + "/" + b;
    }

    public void simplify(){
        int g = gcd(abs(a), abs(b));
        this.a = a / g;
        this.b = b / g;
    }

    private int abs (int x) { return x < 0 ? -x : x; }
    // ========================================== Do not change the methods below.

    private int getNumerator() {
        return a;
    }

    private void setNumerator(int a) {
        this.a = a;
    }

    private int getDenominator() {
        return b;
    }

    private void setDenominator(int b) {
        this.b = b;
    }

    private int gcd(int a, int b){
        while (b != 0) {
            int t = b;
            b = a % b;
            a = t;
        }
        return a;
    }
}
