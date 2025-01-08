package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when a directory or document of the same name  already exists in the directory.
 * Note: a name of a document excludes the file extension.
 */
public class FileExistsException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public FileExistsException(String message) {
        super(message);
    }
}
