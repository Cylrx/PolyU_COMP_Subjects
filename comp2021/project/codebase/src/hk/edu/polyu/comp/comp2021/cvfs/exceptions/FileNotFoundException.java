package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when neither a document nor a directory is found in the directory.
 * Note: a name of a document excludes the file extension.
 */
public class FileNotFoundException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public FileNotFoundException(String message) {
        super(message);
    }
}
