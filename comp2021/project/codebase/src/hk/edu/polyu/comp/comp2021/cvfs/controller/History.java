package hk.edu.polyu.comp.comp2021.cvfs.controller;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.CVFSException;
import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidActionException;

import java.util.Stack;

/**
 * History class to keep track of all the actions performed by the user.
 * It is responsible for undoing and redoing actions.
 */
public class History {
    private final Stack<Instruction> undoStack;
    private final Stack<Instruction> redoStack;

    /**
     * Constructor for History class.
     * Initializes the undo stack and redo stack.
     */
    History() {
        undoStack = new Stack<>();
        redoStack = new Stack<>();
    }

    /**
     * Clears both undo and redo stack
     */
    protected void clear() {
        undoStack.clear();
        redoStack.clear();
    }

    /**
     * Undo the last action performed by the user.
     * @throws CVFSException when there are no actions to undo or when failed to undo the action due to various reasons specific to that instruction.
     */
    protected void undo() throws CVFSException {
        assertNotEmpty(undoStack);
        Instruction inst = undoStack.pop();
        inst.reverse();
        redoStack.push(inst);
    }

    /**
     * Redo the last action that was undone by the user.
     * This action can only be performed after an undo action and before any new action is performed.
     * @throws CVFSException when there are no actions to redo or when failed to redo the action due to various reasons specific to that instruction.
     */
    protected void redo() throws CVFSException {
        assertNotEmpty(redoStack);
        Instruction inst = redoStack.pop();
        inst.execute();
        undoStack.push(inst);
    }

    /**
     * @param inst the instruction to be executed
     * @throws CVFSException when failed to execute the instruction due to various reasons specific to that instruction.
     */
    void run(Instruction inst) throws CVFSException {
        inst.execute();
        undoStack.push(inst);
        redoStack.clear();
    }

    private void assertNotEmpty(Stack<Instruction> stack) throws InvalidActionException {
        if (!stack.isEmpty()) return;
        if (stack == undoStack) throw new InvalidActionException("cannot undo: no action to undo");
        if (stack == redoStack) throw new InvalidActionException("cannot redo: no action to redo");
    }
}
