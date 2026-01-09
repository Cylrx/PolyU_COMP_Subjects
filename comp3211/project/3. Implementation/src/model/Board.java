package model;

import model.elements.Coordinate;
import model.elements.PieceType;
import model.elements.SquareType;
import model.logging.*;

import java.io.Serializable;

public class Board implements Serializable {
    private int length, width;
    private Square[][] board;
    private Coordinate lowerPlayerDen, upperPlayerDen;
    private final MovingValidator movingValidator;
    private Logger logger;


    public Board() {
        setDefaultBoard();
        movingValidator = new MovingValidator(this);
        logger = new Logger();
    }

    private void setDefaultBoard() {
        length = 7;
        width = 9;
        board = new Square[length + 1][width + 1];

        // Set up special square
        board[3][1] = new SpecialSquare(new Coordinate(3, 1), SquareType.Trap, false);
        board[4][1] = new SpecialSquare(new Coordinate(4, 1), SquareType.Den, false);
        lowerPlayerDen = new Coordinate(4, 1);
        board[5][1] = new SpecialSquare(new Coordinate(5, 1), SquareType.Trap, false);
        board[4][2] = new SpecialSquare(new Coordinate(4, 2), SquareType.Trap, false);

        board[3][9] = new SpecialSquare(new Coordinate(3, 9), SquareType.Trap, true);
        board[4][9] = new SpecialSquare(new Coordinate(4, 9), SquareType.Den, true);
        upperPlayerDen = new Coordinate(4, 9);
        board[5][9] = new SpecialSquare(new Coordinate(5, 9), SquareType.Trap, true);
        board[4][8] = new SpecialSquare(new Coordinate(4, 8), SquareType.Trap, true);

        // Set up river
        board[2][4] = new Square(new Coordinate(2, 4), SquareType.River);
        board[3][4] = new Square(new Coordinate(3, 4), SquareType.River);
        board[2][5] = new Square(new Coordinate(2, 5), SquareType.River);
        board[3][5] = new Square(new Coordinate(3, 5), SquareType.River);
        board[2][6] = new Square(new Coordinate(2, 6), SquareType.River);
        board[3][6] = new Square(new Coordinate(3, 6), SquareType.River);

        board[5][4] = new Square(new Coordinate(5, 4), SquareType.River);
        board[6][4] = new Square(new Coordinate(6, 4), SquareType.River);
        board[5][5] = new Square(new Coordinate(5, 5), SquareType.River);
        board[6][5] = new Square(new Coordinate(6, 5), SquareType.River);
        board[5][6] = new Square(new Coordinate(5, 6), SquareType.River);
        board[6][6] = new Square(new Coordinate(6, 6), SquareType.River);

        // Set up normal squares
        for (int i = 1; i <= length; i++) {
            for (int j = 1; j <= width; j++) {
                if (board[i][j] == null) {
                    board[i][j] = new Square(new Coordinate(i, j), SquareType.Normal);
                }
            }
        }

        // Set up pieces
        board[7][3].setPiece(new Piece(PieceType.Rat, false));
        board[2][2].setPiece(new Piece(PieceType.Cat, false));
        board[6][2].setPiece(new Piece(PieceType.Dog, false));
        board[3][3].setPiece(new Piece(PieceType.Wolf, false));
        board[5][3].setPiece(new Piece(PieceType.Leopard, false));
        board[1][1].setPiece(new Piece(PieceType.Tiger, false));
        board[7][1].setPiece(new Piece(PieceType.Lion, false));
        board[1][3].setPiece(new Piece(PieceType.Elephant, false));

        board[1][7].setPiece(new Piece(PieceType.Rat, true));
        board[6][8].setPiece(new Piece(PieceType.Cat, true));
        board[2][8].setPiece(new Piece(PieceType.Dog, true));
        board[5][7].setPiece(new Piece(PieceType.Wolf, true));
        board[3][7].setPiece(new Piece(PieceType.Leopard, true));
        board[7][9].setPiece(new Piece(PieceType.Tiger, true));
        board[1][9].setPiece(new Piece(PieceType.Lion, true));
        board[7][7].setPiece(new Piece(PieceType.Elephant, true));
    }

    public Piece getPiece(int x, int y) {
        if(x <= 0 || x > length || y <= 0 || y > width) {
            throw new IndexOutOfBoundsException();
        }
        return board[x][y].getPiece();
    }


    public int getLength() {
        return length;
    }

    public int getWidth() {
        return width;
    }

    public Square getSquare(Coordinate c) {
        if(!c.checkBound(getLength(), getWidth())) {
            throw new IndexOutOfBoundsException();
        }
        return board[c.getX()][c.getY()];
    }

    public boolean attemptMove(String name, boolean turn, char direction) {
        Coordinate newCoordinate;
        try {
            newCoordinate = movingValidator.attemptMove(name, turn, direction);
        } catch (IllegalArgumentException e) {
            System.out.println(e.getMessage());
            return false;
        }

        Piece curPiece = movingValidator.getTargetPiece(name, turn);
        Square newSquare = getSquare(newCoordinate), originalSquare = curPiece.getPosition();

        if(newSquare.getPiece() != null) {
            logger.addCaptured(newSquare.getCoordinate(), newSquare.getPiece(), this);
            newSquare.getPiece().setDie();
            newSquare.setPiece(null);
        }

        logger.addMove(originalSquare.getCoordinate(), curPiece, this, true, newSquare.getCoordinate());

        originalSquare.setPiece(null);
        newSquare.setPiece(curPiece);

        return true;
    }

    public void tryWithdraw(boolean turn) {
        logger.tryWithdraw(turn);
    }

    public Logger getLogger() {
        return logger;
    }

    public boolean checkDen(boolean turn) {
        if(turn && getSquare(upperPlayerDen).getPiece() != null) {
            return true;
        }
        return !turn && getSquare(lowerPlayerDen).getPiece() != null;
    }

    public boolean haveValidMove(boolean turn) {
        return movingValidator.haveValidMove(turn);
    }

    // for test only
    public MovingValidator getMovingValidator() {
        return movingValidator;
    }
}
