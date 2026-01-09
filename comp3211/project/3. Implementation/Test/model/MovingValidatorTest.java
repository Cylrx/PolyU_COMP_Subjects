package model;

import model.elements.Coordinate;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MovingValidatorTest {
    private static Board board;
    private static MovingValidator movingValidator;

    @BeforeEach
    public void init() {
        board = new Board();
        movingValidator = board.getMovingValidator();
    }

    @Test
    public void testSize() {
        assertEquals(8, movingValidator.getUpperDictionary().size());
        assertEquals(8, movingValidator.getLowerDictionary().size());
    }

    @Test
    public void testMove() {
        try {
            movingValidator.attemptMove("wolf", false, 'U');
            fail();
        } catch (Exception e) {
            System.out.println(e.getMessage());
        }
        assertEquals(movingValidator.attemptMove("elephant", false, 'U'), new Coordinate(1, 4));
        assertEquals(movingValidator.attemptMove("elephant", false, 'R'), new Coordinate(2, 3));
    }

    @Test
    public void testCapture() {
        Piece a = movingValidator.getTargetPiece("elephant", false), b = movingValidator.getTargetPiece("wolf", false);
        assertFalse(movingValidator.attemptCapture(a, b));

        b = movingValidator.getTargetPiece("wolf", true);
        assertTrue(movingValidator.attemptCapture(a, b));

        b = movingValidator.getTargetPiece("elephant", true);
        assertTrue(movingValidator.attemptCapture(a, b));

        a = movingValidator.getTargetPiece("rat", false);
        b = movingValidator.getTargetPiece("elephant", true);
        assertTrue(movingValidator.attemptCapture(a, b));

        a.setPosition(board.getSquare(new Coordinate(2, 4)));

        b = movingValidator.getTargetPiece("rat", true);
        assertFalse(movingValidator.attemptCapture(a, b));

        b.setPosition(board.getSquare(new Coordinate(2, 5)));
        assertTrue(movingValidator.attemptCapture(a, b));

        b.setPosition(board.getSquare(new Coordinate(3,1)));
        a = movingValidator.getTargetPiece("elephant", false);
        assertTrue(movingValidator.attemptCapture(a, b));

        b = movingValidator.getTargetPiece("elephant", true);
        b.setPosition(board.getSquare(new Coordinate(3,1)));

        a = movingValidator.getTargetPiece("wolf", false);
        assertTrue(movingValidator.attemptCapture(a, b));

    }

    @Test
    public void testRiverPieceCannotCaptureLandPiece() {
        board.attemptMove("rat", false, 'L');
        board.attemptMove("rat", false, 'U');
        board.attemptMove("rat", false, 'U');
        board.attemptMove("rat", false, 'U');
        
        Piece ratInRiver = movingValidator.getTargetPiece("rat", false);
        Piece elephantOnLand = movingValidator.getTargetPiece("elephant", true);
        
        assertFalse(movingValidator.attemptCapture(ratInRiver, elephantOnLand));
    }

    @Test
    public void testLandPieceCannotCaptureRiverPiece() {
        board.attemptMove("rat", false, 'L');
        board.attemptMove("rat", false, 'U');
        board.attemptMove("rat", false, 'U');
        board.attemptMove("rat", false, 'U');
        
        Piece ratInRiver = movingValidator.getTargetPiece("rat", false);
        Piece elephantOnLand = movingValidator.getTargetPiece("elephant", true);
        
        assertFalse(movingValidator.attemptCapture(elephantOnLand, ratInRiver));
    }


    @Test
    public void testRiverPieceCanCaptureRiverPiece() {
        board.attemptMove("rat", false, 'L');
        board.attemptMove("rat", false, 'U');
        board.attemptMove("rat", false, 'U');
        board.attemptMove("rat", false, 'U');
        
        board.attemptMove("rat", true, 'R');
        board.attemptMove("rat", true, 'D');
        board.attemptMove("rat", true, 'D');
        board.attemptMove("rat", true, 'D');

        
        Piece rat1 = movingValidator.getTargetPiece("rat", false);
        Piece rat2 = movingValidator.getTargetPiece("rat", true);
        
        assertTrue(movingValidator.attemptCapture(rat1, rat2));
    }

}