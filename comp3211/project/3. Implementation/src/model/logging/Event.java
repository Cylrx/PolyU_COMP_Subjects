package model.logging;

import model.Piece;
import model.Board;
import model.elements.Coordinate;

import java.io.Serializable;

public abstract class Event implements Serializable {
    private final EventType type;
    public Event(EventType type) {
        this.type = type;
    }

    public abstract void printMessage();
    public abstract void printReverseMessage();
}

abstract class MoveEvent extends Event {
    private final Coordinate originalCoordinate;
    private final Piece targetPiece;
    private final Board belBoard;

    protected MoveEvent(Coordinate _originalCoordinate, Piece _targetPiece, Board _belBoard, EventType _type) {
        super(_type);
        originalCoordinate = _originalCoordinate;
        targetPiece = _targetPiece;
        belBoard = _belBoard;
    }

    public Coordinate getOriginalCoordinate() {
        return originalCoordinate;
    }

    public Piece getTargetPiece() {
        return targetPiece;
    }

    public Board getBoard() {
        return belBoard;
    }

    public abstract void withdraw();
    public abstract void run();

}

class Captured extends MoveEvent {
    public Captured(Coordinate _originalCoordinate, Piece _targetPiece, Board _belBoard) {
        super(_originalCoordinate, _targetPiece, _belBoard, EventType.Captured);
    }
    public void withdraw() {
        Board curBoard = super.getBoard();
        super.getTargetPiece().setAlive();
        curBoard.getSquare(super.getOriginalCoordinate()).setPiece(super.getTargetPiece());
    }
    public void run() {
        Board curBoard = super.getBoard();
        curBoard.getSquare(super.getOriginalCoordinate()).setPiece(null);
        super.getTargetPiece().setDie();
    }
    public void printMessage() {
        System.out.println("- " + getOriginalCoordinate() + " " + getTargetPiece().getType());
    }
    public void printReverseMessage() {
        System.out.println("+ " + getOriginalCoordinate() + " " + getTargetPiece().getType());
    }
}

class Move extends MoveEvent { // True for success
    private final Coordinate targetCoordinate;
    public Move(Coordinate _originalCoordinate, Piece _targetPiece, Board _belBoard, Coordinate _targetCoordinate) {
        super(_originalCoordinate, _targetPiece, _belBoard, EventType.Move);
        targetCoordinate = _targetCoordinate;
    }
    public void withdraw() {
        Board curBoard = super.getBoard();
        curBoard.getSquare(targetCoordinate).setPiece(null);
        curBoard.getSquare(super.getOriginalCoordinate()).setPiece(super.getTargetPiece());
    }
    public void run() {
        Board curBoard = super.getBoard();
        curBoard.getSquare(super.getOriginalCoordinate()).setPiece(null);
        curBoard.getSquare(targetCoordinate).setPiece(super.getTargetPiece());
    }
    public void printMessage() {
        System.out.println(getTargetPiece().getType() + " : " + getOriginalCoordinate() + " -> " + targetCoordinate);
    }
    public void printReverseMessage() {
        System.out.println(getTargetPiece().getType() + " : " + targetCoordinate + " -> " + getOriginalCoordinate());
    }
}

class Withdraw extends Event {
    private final boolean turn;
    public Withdraw(boolean _turn) {
        super(EventType.Withdraw);
        turn = _turn;
    }

    public void printMessage() {
        System.out.println(turn ? "upperPlayer" : "lowerPlayer" + " +Withdraw");
    }
    public void printReverseMessage() {
        System.out.println(turn ? "upperPlayer" : "lowerPlayer" + " -Withdraw");
    }
}