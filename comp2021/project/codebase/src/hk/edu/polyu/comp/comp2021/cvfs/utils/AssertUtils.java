package hk.edu.polyu.comp.comp2021.cvfs.utils;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.*;
import hk.edu.polyu.comp.comp2021.cvfs.model.criterion.Criterion;

import static hk.edu.polyu.comp.comp2021.cvfs.utils.TypeUtils.*;

/**
 * A utility class that provides assertion methods for various validity checking
 */
public class AssertUtils {
    /**
     * Asserts that the given arguments == the expected count
     * If the assertion fails, format an error message that indicates the mismatch
     * @param commandName the name of the command (for error message formatting purpose only)
     * @param args the array of arguments to check
     * @param expectedCount the expected number of arguments
     * @throws InvalidCommandException if the number of arguments does not match the expected count
     */
    public static void assertArgumentCount(String commandName, String[] args, int expectedCount) throws InvalidCommandException {
        if (args.length != expectedCount){
            String errorMessage = String.format(
                    "command \"%s\": expected %d arguments but received %d",
                    commandName, expectedCount, args.length
            );
            throw new InvalidCommandException(errorMessage);
        }
    }

    /**
     * Asserts that the given criterion name is valid (must be exactly 2 english letters)
     * If the assertion fails, format an error message that indicates the error
     * @param crtName the criterion name to check
     * @throws InvalidCriterionException if the criterion name is invalid
     */
    public static void assertValidCrtName (String crtName) throws InvalidCriterionException {
        if (crtName.length() != 2) throw Criterion.getInvalidCriterionException("criterion name", "exactly 2 letters", crtName);
        if (!isAlpha(crtName)) throw Criterion.getInvalidCriterionException("criterion name", "english letters", crtName);
    }

    /**
     * Asserts that the given String is numeric
     * If the assertion fails, format an error message that indicates the error
     * @param reason the command that has an invalid numeric argument (for error message formatting only)
     * @param val the String value to check
     * @throws InvalidValueException when the value is not numeric
     */
    public static void assertNumber (String reason, String val) throws InvalidValueException {
        if (isNumber(val)) return;
        String errorMessage = String.format(
                "invalid %s: expects integer, but received %s",
                reason, val
        );
        throw new InvalidValueException(errorMessage);
    }

    /**
     * Asserts that the numeric value of the given String <code>val</code> is within <code>limit</code>
     * If the assertion fails, format an error message that indicates the error.
     * @param reason the reason for the limit check (for error message formatting only)
     * @param val the String value to check
     * @param limit the upper limit of the value <b>(inclusive)</b>
     * @throws InvalidValueException when the value is greater than the limit
     */
    public static void assertLimit (String reason, String val, long limit) throws InvalidValueException {
        String lim = Long.toString(limit);
        int nv = val.length();
        int nl = lim.length();
        if (nv < nl) return;
        if (nv == nl) {
            int i = 0;
            while(i < nl && val.charAt(i) == lim.charAt(i)) i++;
            if (i == nl || val.charAt(i) <= lim.charAt(i)) return;
        }
        String errorMessage = String.format(
                "invalid %s: expects < %d, but received %s",
                reason, limit, val
        );
        throw new InvalidValueException(errorMessage);
    }

    /**
     * Asserts that the given document name is valid (must be alphanumeric and ≤ 10 characters)
     * If the assertion fails, format an error message that indicates the error
     * @param component the component that has an invalid name (for error message formatting only)
     * @param docName the document name to check
     * @throws InvalidFileException when the document name is invalid
     */
    public static void assertValidFileName(String component, String docName) throws InvalidFileException {
        if (isAlphaNumeric(docName) || docName.length() > 10) {
            String errorMessage = String.format(
                    "invalid %s: expects alphanumeric, length <= 10",
                    component
            );
            throw new InvalidFileException(errorMessage);
        }
    }
}
