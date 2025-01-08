package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Superclass exception for all exceptions in CVFS.
 * All custom checked exceptions in CVFS should extend this class.
 */
public class CVFSException extends Exception {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public CVFSException(String message) { super(message); }
}
