package model;

import model.elements.Coordinate;
import model.elements.SquareType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class SpecialSquareTest {
    private SpecialSquare trapSquare;
    private SpecialSquare denSquare;
    private Coordinate coordinate;

    @BeforeEach
    public void setUp() {
        coordinate = new Coordinate(3, 1);
        trapSquare = new SpecialSquare(coordinate, SquareType.Trap, false);
        denSquare = new SpecialSquare(new Coordinate(4, 1), SquareType.Den, false);
    }

    @Test
    public void testConstructor() {
        assertNotNull(trapSquare);
        assertEquals(coordinate, trapSquare.getCoordinate());
        assertEquals(SquareType.Trap, trapSquare.getType());
        assertFalse(trapSquare.getSide());
    }

    @Test
    public void testGetSide() {
        assertFalse(trapSquare.getSide());
        
        SpecialSquare upperTrap = new SpecialSquare(new Coordinate(3, 9), SquareType.Trap, true);
        assertTrue(upperTrap.getSide());
    }

    @Test
    public void testInheritance() {
        Piece piece = new Piece(model.elements.PieceType.Elephant, false);
        trapSquare.setPiece(piece);
        assertEquals(piece, trapSquare.getPiece());
        assertEquals(trapSquare, piece.getPosition());
    }

    @Test
    public void testDenSquare() {
        assertEquals(SquareType.Den, denSquare.getType());
        assertFalse(denSquare.getSide());
        
        SpecialSquare upperDen = new SpecialSquare(new Coordinate(4, 9), SquareType.Den, true);
        assertTrue(upperDen.getSide());
    }
}

