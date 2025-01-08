package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when the remaining space in the disk is insufficient to store the new document or directory.
 * It does not necessarily indicate that the disk is fully occupied.
 */
public class FullDiskException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public FullDiskException(String message) {
        super(message);
    }
}
