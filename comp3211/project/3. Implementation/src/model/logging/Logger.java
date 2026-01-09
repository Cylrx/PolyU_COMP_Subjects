package model.logging;

import model.elements.Coordinate;
import model.Board;
import model.Piece;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Stack;

public class Logger implements Serializable {
    private final Stack<MoveEvent> curStack;
    private final ArrayList<Event> allEvents;
    private int moveCounter;

    // For replay
    private Stack<MoveEvent> reverseStack;
    private int curPointer;

    public Logger() {
        curStack = new Stack<>();
        allEvents = new ArrayList<>();
        moveCounter = 0;
    }

    private void addEvent(Event event) {
        allEvents.add(event);
        if(event instanceof MoveEvent) {
            curStack.push((MoveEvent) event);
        }
        if(event instanceof Move) {
            moveCounter++;
        }
    }

    public void addCaptured(Coordinate _originalCoordinate, Piece _targetPiece, Board _belBoard) {
        Captured curEvent = new Captured(_originalCoordinate, _targetPiece, _belBoard);
        addEvent(curEvent);
        curEvent.printMessage();
    }

    public void addMove(Coordinate _originalCoordinate, Piece _targetPiece, Board _belBoard, boolean _success, Coordinate _targetCoordinate) {
        Move moveEvent = new Move(_originalCoordinate, _targetPiece, _belBoard, _targetCoordinate);
        addEvent(moveEvent);
        moveEvent.printMessage();
    }

    public void tryWithdraw(boolean turn) {
        if(moveCounter < 2) {
            throw new IllegalArgumentException("You did not made any move");
        }
        int rem = 2;
        while(rem > 0) {
            MoveEvent currentEvent = curStack.pop();
            if(currentEvent instanceof Move) {
                moveCounter--;
                rem--;
            }
            currentEvent.withdraw();
            currentEvent.printReverseMessage();
            if(rem == 0 && !curStack.empty() && curStack.peek() instanceof Captured) {
                currentEvent = curStack.pop();
                currentEvent.withdraw();
                currentEvent.printReverseMessage();
            }
        }
        Withdraw withdrawEvent = new Withdraw(turn);
        addEvent(withdrawEvent);
        withdrawEvent.printMessage();
    }

    // For replay
    public void initReplay() {
        reverseStack = new Stack<>();
        curPointer = 0;
        while(!curStack.isEmpty()) {
            curStack.pop().withdraw();
        }
    }

    // Must call initReplay before use, true for need swap role
    public boolean nextStep() {
        if(curPointer == allEvents.size()) {
            throw new IllegalStateException("No more next step!");
        }
        Event curEvent = allEvents.get(curPointer);
        curEvent.printMessage();
        if(curEvent instanceof Move) {
            curStack.push((MoveEvent) allEvents.get(curPointer));
            ((MoveEvent) curEvent).run();
            curPointer++;
            return true;
        } else if(curEvent instanceof Captured) {
            curStack.push((Captured) allEvents.get(curPointer));
            ((Captured) curEvent).run();
            curPointer++;
            curEvent = allEvents.get(curPointer);
            curEvent.printMessage();
            curStack.push((MoveEvent) allEvents.get(curPointer));
            ((MoveEvent) curEvent).run();
            curPointer++;
            return true;
        } else if(curEvent instanceof Withdraw) {
            int rem = 2;
            while(rem > 0) {
                reverseStack.push(curStack.peek());
                MoveEvent currentEvent = curStack.pop();
                if(currentEvent instanceof Move) {
                    rem--;
                }
                currentEvent.withdraw();
                currentEvent.printReverseMessage();
                if(rem == 0 && !curStack.empty() && curStack.peek() instanceof Captured) {
                    reverseStack.push(curStack.peek());
                    currentEvent = curStack.pop();
                    currentEvent.withdraw();
                    currentEvent.printReverseMessage();
                }
            }
            curPointer++;
            return false;
        }
        return false;
    }

    public boolean previousStep() {
        if(curPointer == 0) {
            throw new IllegalStateException("No more previous step!");
        }
        curPointer--;
        Event curEvent = allEvents.get(curPointer);
        if(curEvent instanceof Move) {
            curStack.pop();
            ((MoveEvent) curEvent).withdraw();
            if(curPointer > 0 && allEvents.get(curPointer - 1) instanceof Captured) {
                curPointer--;
                curStack.pop().withdraw();
            }
            return true;
        } else if(curEvent instanceof Withdraw) {
            curEvent.printReverseMessage();
            int rem = 2;
            while(rem > 0) {
                curStack.push(reverseStack.peek());
                MoveEvent currentEvent = reverseStack.pop();
                if(currentEvent instanceof Move) {
                    rem--;
                }
                currentEvent.run();
            }
            return false;
        }
        return false;
    }

    // for testing only
    public int getStackSize() {
        return curStack.size();
    }
}
