package model;

import model.elements.Coordinate;
import model.elements.SquareType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SquareTest {
    private Square square;
    private Coordinate coordinate;

    @BeforeEach
    public void setUp() {
        coordinate = new Coordinate(3, 4);
        square = new Square(coordinate, SquareType.Normal);
    }

    @Test
    public void testConstructor() {
        assertNotNull(square);
        assertEquals(coordinate, square.getCoordinate());
        assertEquals(SquareType.Normal, square.getType());
        assertNull(square.getPiece());
    }

    @Test
    public void testSetPiece() {
        Piece piece = new Piece(model.elements.PieceType.Rat, false);
        square.setPiece(piece);
        assertEquals(piece, square.getPiece());
        assertEquals(square, piece.getPosition());
    }

    @Test
    public void testSetPieceNull() {
        Piece piece = new Piece(model.elements.PieceType.Cat, true);
        square.setPiece(piece);
        assertNotNull(square.getPiece());
        square.setPiece(null);
        assertNull(square.getPiece());
    }

    @Test
    public void testGetCoordinate() {
        Coordinate newCoord = new Coordinate(5, 6);
        Square newSquare = new Square(newCoord, SquareType.River);
        assertEquals(newCoord, newSquare.getCoordinate());
    }

    @Test
    public void testGetType() {
        Square riverSquare = new Square(new Coordinate(2, 4), SquareType.River);
        assertEquals(SquareType.River, riverSquare.getType());
        
        Square trapSquare = new Square(new Coordinate(3, 1), SquareType.Trap);
        assertEquals(SquareType.Trap, trapSquare.getType());
    }
}

