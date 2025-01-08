package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when an invalid file type is entered.
 * Valid file types are {txt, java, html, css}
 */
public class InvalidTypeException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public InvalidTypeException(String message) {
        super(message);
    }
}
