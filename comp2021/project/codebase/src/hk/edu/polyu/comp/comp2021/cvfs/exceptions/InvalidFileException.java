package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when the file name is invalid.
 */
public class InvalidFileException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public InvalidFileException(String message) {
        super(message);
    }
}
