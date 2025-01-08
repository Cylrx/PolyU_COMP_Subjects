package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when an invalid IO operation is attempted.
 * Primarily used when serializing the virtual disk.
 * Example use case:
 * - Attempting to read / write to a path that does not exist
 * - Attempting to read / write to a path that is has no permissions
 */
public class InvalidIOException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public InvalidIOException(String message) {
        super(message);
    }
}
