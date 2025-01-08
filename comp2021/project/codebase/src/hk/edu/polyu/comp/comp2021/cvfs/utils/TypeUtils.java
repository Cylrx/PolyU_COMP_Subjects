package hk.edu.polyu.comp.comp2021.cvfs.utils;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidValueException;

import static hk.edu.polyu.comp.comp2021.cvfs.utils.AssertUtils.assertLimit;

/**
 * A utility class that provides methods for type checking and conversion
 */
public class TypeUtils {

    /**
     * Asserts that the given String is numeric
     * @param str the String to check
     * @return true if the String is numeric, false otherwise
     */
    public static boolean isNumber (String str) {
        if (str == null || str.isEmpty()) return false;
        if (str.length() > 1 && str.charAt(0) == '0') return false;
        for (int i = 0; i < str.length(); i++) {
            if (!Character.isDigit(str.charAt(i))) return false;
        }
        return true;
    }

    /**
     * Convert the given String to a number
     * @param str the String to convert
     * @return the number represented by the String
     * @throws InvalidValueException when the String exceeds the limit of long
     */
    public static long toNumber(String str) throws InvalidValueException {
        assertLimit("value conversion", str, Long.MAX_VALUE);
        return Long.parseLong(str);
    }

    /**
     * Asserts that the given String is alphanumeric
     * @param val the String to check
     * @return true if the String is alphanumeric, false otherwise
     */
    public static boolean isAlphaNumeric (String val) {
        if (val == null || val.isEmpty()) return true;
        for (int i = 0; i < val.length(); i++){
            if (!Character.isLetterOrDigit(val.charAt(i))) return true;
        }
        return false;
    }

    /**
     * Asserts that the given String is alphabetic
     * @param str the String to check
     * @return true if the String is alphabetic, false otherwise
     */
    public static boolean isAlpha (String str) {
        if (str == null || str.isEmpty()) return false;
        for (int i = 0; i < str.length(); i++) {
            if (!Character.isLetter(str.charAt(i))) return false;
        }
        return true;
    }
}
