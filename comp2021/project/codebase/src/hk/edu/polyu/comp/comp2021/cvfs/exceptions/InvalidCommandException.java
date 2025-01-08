package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when an invalid command is entered.
 * Command refers to the first word of the input string.
 * Command does not include the subsequent arguments.
 */
public class InvalidCommandException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public InvalidCommandException(String message) { super(message); }
}
