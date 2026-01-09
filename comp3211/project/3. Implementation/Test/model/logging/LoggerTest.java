package model.logging;

import model.Board;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class LoggerTest {
    private Board board;
    private Logger logger;

    @BeforeEach
    public void setUp() {
        board = new Board();
        logger = board.getLogger();
    }

    @Test
    public void testAddMove() {
        board.attemptMove("rat", false, 'L');
        assertNotNull(logger);
    }

    @Test
    public void testAddCaptured() {
        board.attemptMove("elephant", false, 'U');
        board.attemptMove("elephant", false, 'U');
        board.attemptMove("elephant", false, 'U');
        
        board.attemptMove("elephant", false, 'R');
        assertNotNull(logger);
    }

    @Test
    public void testTryWithdraw() {
        board.attemptMove("rat", false, 'L');
        board.attemptMove("cat", true, 'D');
        
        assertDoesNotThrow(() -> board.tryWithdraw(false));
    }

    @Test
    public void testTryWithdrawNotEnoughMoves() {
        board.attemptMove("rat", false, 'L');
        
        assertThrows(IllegalArgumentException.class, () -> {
            board.tryWithdraw(false);
        });
    }

    @Test
    public void testInitReplay() {
        board.attemptMove("rat", false, 'L');
        board.attemptMove("cat", false, 'U');
        
        assertDoesNotThrow(() -> {
            logger.initReplay();
        });
    }

    @Test
    public void testNextStep() {
        board.attemptMove("rat", false, 'L');
        board.attemptMove("cat", false, 'U');
        
        logger.initReplay();
        
        assertDoesNotThrow(() -> {
            boolean result = logger.nextStep();
            assertTrue(result);
        });
    }

    @Test
    public void testNextStepLastStep() {
        board.attemptMove("rat", false, 'L');
        
        logger.initReplay();
        logger.nextStep();
        
        assertThrows(IllegalStateException.class, () -> {
            logger.nextStep();
        });
    }

    @Test
    public void testPreviousStep() {
        board.attemptMove("leopard", false, 'L');
        board.attemptMove("rat", true, 'D');
        board.attemptMove("leopard", false, 'U');
        board.attemptMove("rat", true, 'D');
        board.attemptMove("leopard", false, 'U');
        board.attemptMove("rat", true, 'D');
        assertDoesNotThrow(() -> {
            board.tryWithdraw(false);
        });
        board.attemptMove("leopard", false, 'U');
        board.attemptMove("rat", true, 'D');
        board.attemptMove("leopard", false, 'U');
        board.attemptMove("cat", true, 'L');
        board.attemptMove("leopard", false, 'U');
        board.attemptMove("cat", true, 'L');
        board.attemptMove("leopard", false, 'U');
        assertDoesNotThrow(() -> {
            board.tryWithdraw(true);
        });
        board.attemptMove("cat", true, 'L');
        board.attemptMove("leopard", false, 'U');
        board.attemptMove("dog", true, 'D');
        assertDoesNotThrow(() -> {
            board.tryWithdraw(false);
        });
        board.attemptMove("leopard", false, 'U');
        board.attemptMove("dog", true, 'D');
        board.attemptMove("leopard", false, 'U');


        logger.initReplay();

        for(int i = 0; i < 22; i++) {
            assertDoesNotThrow(() -> {
                logger.nextStep();
            });
        }


        assertThrows(Exception.class, () -> {
            logger.nextStep();
        });

        for(int i = 0; i < 22; i++) {
            assertDoesNotThrow(() -> {
                logger.previousStep();
            });
        }

    }

    @Test
    public void testPreviousStepAtStart() {
        board.attemptMove("rat", false, 'L');
        
        logger.initReplay();
        
        assertThrows(IllegalStateException.class, () -> {
            logger.previousStep();
        });
    }
}

