package model;

import model.elements.Coordinate;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class BoardTest {
    private Board board;

    @BeforeEach
    public void setUp() {
        board = new Board();
    }

    @Test
    public void testMoving1() {
        // normal move exceed border
        assertFalse(board.attemptMove("tiger", false, 'D'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertFalse(board.attemptMove("tiger", false, 'U'));

        // normal move exceed border
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertFalse(board.attemptMove("elephant", false, 'R'));

        // normal move into den
        assertTrue(board.attemptMove("lion", false, 'L'));
        assertTrue(board.attemptMove("lion", false, 'L'));
        assertFalse(board.attemptMove("lion", false, 'L'));
    }


    @Test
    public void waterAttack() {
        // attack from water
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", true, 'D'));
        assertTrue(board.attemptMove("rat", true, 'R'));
        assertTrue(board.attemptMove("rat", true, 'R'));
        assertTrue(board.attemptMove("rat", true, 'D'));
        assertTrue(board.attemptMove("rat", true, 'D'));
    }

    @Test
    public void testMoving2() {
        // block by an enemy piece
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertFalse(board.attemptMove("elephant", false, 'U'));

        // jump across river
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'R'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
    }

    @Test
    public void testMoving3() {
        // attack from water
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("elephant", true, 'L'));
        assertFalse(board.attemptMove("rat", false, 'U'));
    }

    @Test
    public void testMoving4() {
        // get into other's den
        assertTrue(board.attemptMove("leopard", false, 'L'));
        assertTrue(board.attemptMove("leopard", false, 'U'));
        assertTrue(board.attemptMove("leopard", false, 'U'));
        assertTrue(board.attemptMove("leopard", false, 'U'));
        assertTrue(board.attemptMove("leopard", false, 'U'));
        assertTrue(board.attemptMove("leopard", false, 'U'));
        assertTrue(board.attemptMove("leopard", false, 'U'));
    }

    @Test
    public void testOnlyRatCanEnterRiver() {
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        
        assertTrue(board.attemptMove("cat", false, 'U'));
        assertFalse(board.attemptMove("cat", false, 'U'));

        assertTrue(board.attemptMove("dog", false, 'U'));
        assertFalse(board.attemptMove("dog", false, 'U'));
        assertFalse(board.attemptMove("elephant", false, 'L'));
    }

    @Test
    public void testRiverPieceCannotAttackLandPiece() {
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("elephant", true, 'L'));
        assertFalse(board.attemptMove("rat", false, 'U'));
    }

    @Test
    public void testEnemyTrapRule() {
        assertTrue(board.attemptMove("wolf", true, 'L'));
        assertTrue(board.attemptMove("wolf", true, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));
    }

    @Test
    public void testOwnTrapRule() {
        assertTrue(board.attemptMove("leopard", false, 'L'));
        assertTrue(board.attemptMove("leopard", false, 'D'));
        assertTrue(board.attemptMove("wolf", true, 'L'));
        assertTrue(board.attemptMove("wolf", true, 'D'));
        assertTrue(board.attemptMove("wolf", true, 'D'));
        assertTrue(board.attemptMove("wolf", true, 'D'));
        assertTrue(board.attemptMove("wolf", true, 'D'));
        assertFalse(board.attemptMove("wolf", true, 'D'));
    }

    @Test
    public void testTigerCanJumpOverRiver() {
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'R'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
    }

    @Test
    public void testLionCanJumpOverRiver() {
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("lion", false, 'U'));
        assertTrue(board.attemptMove("lion", false, 'U'));
        assertTrue(board.attemptMove("lion", false, 'L'));
        assertTrue(board.attemptMove("lion", false, 'U'));

    }

    @Test
    public void testTigerCannotJumpOverRiverWithRat() {
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));

        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'R'));
        assertFalse(board.attemptMove("tiger", false, 'U'));
    }

    @Test
    public void testLionCannotJumpOverRiverWithRat() {
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'U'));

        assertTrue(board.attemptMove("lion", false, 'U'));
        assertTrue(board.attemptMove("lion", false, 'U'));
        assertTrue(board.attemptMove("lion", false, 'L'));
        assertFalse(board.attemptMove("lion", false, 'U'));
    }

    @Test
    public void testCheckDen() {
        assertFalse(board.checkDen(false));
        assertFalse(board.checkDen(true));
        assertTrue(board.attemptMove("wolf", false, 'R'));
        assertTrue(board.attemptMove("wolf", false, 'U'));
        assertTrue(board.attemptMove("wolf", false, 'U'));
        assertTrue(board.attemptMove("wolf", false, 'U'));
        assertTrue(board.attemptMove("wolf", false, 'U'));
        assertTrue(board.attemptMove("wolf", false, 'U'));
        assertTrue(board.attemptMove("wolf", false, 'U'));
        assertTrue(board.checkDen(true));
    }

    @Test
    public void testJumpKill() {
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'R'));
        assertTrue(board.attemptMove("leopard", true, 'L'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertEquals(board.getSquare(new Coordinate(2, 7)).getPiece(), board.getMovingValidator().getTargetPiece("tiger", false));
    }

    @Test
    public void testCannotMove() {
        for(Piece x : board.getMovingValidator().getLowerDictionary().values()) {
            x.setDie();
        }
        assertFalse(board.haveValidMove(false));
    }

    @Test
    public void blockLeap() {
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("tiger", false, 'R'));
        assertTrue(board.attemptMove("rat", true, 'D'));
        assertTrue(board.attemptMove("lion", true, 'D'));
        assertTrue(board.attemptMove("lion", true, 'D'));
        assertTrue(board.attemptMove("lion", true, 'R'));
        assertFalse(board.attemptMove("tiger", false, 'U'));
        assertTrue(board.attemptMove("wolf", false, 'D'));
        assertTrue(board.attemptMove("tiger", false, 'R'));
        assertTrue(board.attemptMove("tiger", false, 'R'));
        assertTrue(board.attemptMove("tiger", false, 'U'));
        assertFalse(board.attemptMove("tiger", false, 'L'));
    }

    @Test
    public void testOutOccupied() {
        // try to get out of the water but occupied by another piece
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("dog", false, 'U'));
        assertFalse(board.attemptMove("rat", false, 'D'));

        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("rat", false, 'L'));

        assertTrue(board.attemptMove("rat", false, 'U'));
        assertTrue(board.attemptMove("rat", false, 'U'));

        assertTrue(board.attemptMove("rat", true, 'R'));
        assertFalse(board.attemptMove("rat", true, 'D'));

    }

    @Test
    public void testGetPiece() {
        assertNotNull(board.getPiece(7, 3));
        assertEquals(model.elements.PieceType.Rat, board.getPiece(7, 3).getType());
        
        assertThrows(IndexOutOfBoundsException.class, () -> {
            board.getPiece(0, 1);
        });
        
        assertThrows(IndexOutOfBoundsException.class, () -> {
            board.getPiece(10, 1);
        });
    }

    @Test
    public void testGetSquare() {
        model.elements.Coordinate coord = new model.elements.Coordinate(1, 1);
        assertNotNull(board.getSquare(coord));
        
        model.elements.Coordinate invalidCoord = new model.elements.Coordinate(10, 10);
        assertThrows(IndexOutOfBoundsException.class, () -> {
            board.getSquare(invalidCoord);
        });
    }

    @Test
    public void testHaveValidMove() {
        assertTrue(board.haveValidMove(false));
        assertTrue(board.haveValidMove(true));
    }

    @Test
    public void testWithdraw() {
        assertTrue(board.attemptMove("wolf", false, 'R'));
        assertTrue(board.attemptMove("wolf", true, 'L'));
        assertTrue(board.attemptMove("wolf", false, 'U'));
        assertTrue(board.attemptMove("wolf", true, 'D'));
        assertTrue(board.attemptMove("wolf", false, 'U'));
        assertTrue(board.attemptMove("wolf", true, 'D'));
        Coordinate central = new Coordinate(4, 5), up = new Coordinate(4, 6), down = new Coordinate(4, 4);
        assertNotNull(board.getSquare(central).getPiece());
        assertNull(board.getSquare(up).getPiece());
        assertNull(board.getSquare(down).getPiece());
        assertDoesNotThrow(() -> {
            board.tryWithdraw(false);
        });
        assertNull(board.getSquare(central).getPiece());
        assertNotNull(board.getSquare(up).getPiece());
        assertNotNull(board.getSquare(down).getPiece());
    }

}