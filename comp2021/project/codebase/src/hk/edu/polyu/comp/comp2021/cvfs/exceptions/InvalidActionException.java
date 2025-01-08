package hk.edu.polyu.comp.comp2021.cvfs.exceptions;

/**
 * Exception thrown when a more miscellaneous action is attempted.
 * Throw this exception when the action does not fall under any other CVFS exceptions.
 * Examples:
 *  - Attempting to undo, when undoStack is empty
 *  - Attempting to redo, when redoStack is empty
 *  - Attempting to create document or directory when no disk is loaded
 */
public class InvalidActionException extends CVFSException {
    /**
     * @param message the message to be displayed when the exception is thrown
     */
    public InvalidActionException(String message) {
        super(message);
    }
}
