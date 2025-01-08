package hk.edu.polyu.comp.comp2021.assignment1;

import java.lang.Math;

public class TinyFloat {

    public static final int TINY_FLOAT_SIZE = 8;
    public static final int SIGN_POS = 0;
    public static final int EXPONENT_POS = 1;
    public static final int MANTISSA_POS = 5;

    public static void main(String[] args){
        System.out.println(numberOfIntegers());
    }

    private static int fast2Pow(int b) {
        return (b >= 0) ? 1 << b : - (1 << -b);
    }

    private static double floor(double a) {
        long x = (long) a;
        return (x < a) ? x : x - 1.0;
    }

    // Task 1a: Complete the method binary2Integer
    // to convert the string value to integer value for the exponent.
    private static int binary2Integer(String exponentString){
        int n = exponentString.length();
        int res = (exponentString.charAt(0) == '0') ? 0 : -fast2Pow(n - 1);
        int pow2 = 1;
        for (int i = n - 1; i >= 1; i--) {
            res += (exponentString.charAt(i) - '0') * pow2;
            pow2 *= 2;
        }

        return res;
    }

    // Task 1b: Complete the method binary2Decimal
    // to convert the string value to float value for the mantissa.
    private static float binary2Decimal(String mantissaString){
        int n = mantissaString.length();
        float pow2 = 0.5F, res = 1F;
        for (int i = 0; i < n; i++) {
            res += (mantissaString.charAt(i) - '0') * pow2;
            pow2 *= 0.5F;
        }
        return res;
    }

    // Task 1c: Complete the method fromString based on the two methods,
    // binary2Integer and binary2Decimal.
    public static float fromString(String bitSequence){
        float mantissa = binary2Decimal(bitSequence.substring(MANTISSA_POS, TINY_FLOAT_SIZE));
        float result = (bitSequence.charAt(SIGN_POS) == '1') ? -mantissa : mantissa;
        int exp = binary2Integer(bitSequence.substring(EXPONENT_POS, MANTISSA_POS));
        float pow = (float)fast2Pow(exp);
        return pow >= 0 ? result * pow : result / -pow;
    }


    public static int numberOfIntegers(){
        // Task 2: return the number of TinyFloat object values that are integers
        int n = 0;
        String[] bitString = getValidTinyFloatBitSequences();
        for (String s: bitString) {
            float res = fromString(s);
            if (res - floor(res) == 0F) {
                System.out.println(s + " == " + (long)Math.floor(res));
                n++;
            }
        }
        return n;
    }

    /**
     * Get all valid bit sequences for tinyFloat values.
     * Do not change the function.
     */
    private static String[] getValidTinyFloatBitSequences(){
        int nbrValues = (int)Math.pow(2, TINY_FLOAT_SIZE);

        String[] result = new String[nbrValues];
        for(int i = 0; i < nbrValues; i++){
            result[i] = String.format("%" + TINY_FLOAT_SIZE + "s", Integer.toBinaryString(i))
                    .replace(' ', '0');
        }
        return result;
    }
}

