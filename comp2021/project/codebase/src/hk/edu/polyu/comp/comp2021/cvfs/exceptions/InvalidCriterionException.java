package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when attempting to create an invalid criterion.
 * Example:
 *  - invalid criterion name
 *  - invalid criterion operand
 *  - invalid criterion operator.
 */
public class InvalidCriterionException extends CVFSException{
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public InvalidCriterionException(String message) {
        super(message);
    }
}
