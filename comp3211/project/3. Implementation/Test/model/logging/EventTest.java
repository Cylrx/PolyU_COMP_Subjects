package model.logging;

import org.junit.jupiter.api.BeforeEach;
import model.Board;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EventTest {
    private Board board;
    private Logger logger;
    
    @BeforeEach
    void setUp() {
        board = new Board();
        logger = board.getLogger();
    }

    @Test
    void testMoveEvent() {
        assertTrue(board.attemptMove("rat", false, 'L'));
        assertNotNull(logger);
    }

    @Test
    void testCapturedEvent() {
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertTrue(board.attemptMove("elephant", false, 'U'));
        assertTrue(board.attemptMove("elephant", false, 'U'));
        
        assertFalse(board.attemptMove("elephant", false, 'R'));
        assertNotNull(logger);
        assertEquals(logger.getStackSize(), 3);
    }

    @Test
    void testEventType() {
        assertEquals("Captured", EventType.Captured.toString());
        assertEquals("Move", EventType.Move.toString());
        assertEquals("Withdraw", EventType.Withdraw.toString());
    }

    @Test
    void testWithdrawEvent() {

        assertTrue(board.attemptMove("rat", false, 'L'));
        assertTrue(board.attemptMove("cat", false, 'U'));
        
        assertDoesNotThrow(() -> board.tryWithdraw(false));
    }
}