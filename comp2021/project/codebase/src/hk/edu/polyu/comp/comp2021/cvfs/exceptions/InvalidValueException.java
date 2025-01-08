package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when an invalid numeric value is entered
 * Example use case:
 *  - Attempting to convert alphabet to number
 *  - Attempting to create number greater than Long.MAX_VALUE
 */
public class InvalidValueException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public InvalidValueException(String message) {
        super(message);
    }
}
