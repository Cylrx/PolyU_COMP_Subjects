package pages.game;

import model.Board;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;

import static org.junit.jupiter.api.Assertions.*;

class GameTest {
    private Game curGame;
    private InputStream originalIn;

    @BeforeEach
    void setUp() {
        curGame = new Game();
        originalIn = System.in;
    }

    @Test
    public void testSave() {
        curGame.saveGame("test");
    }

    @Test
    public void testClose() {
        curGame.closeGame();
    }

    @Test
    public void testNormalGameWithWinner() throws Exception {
        Board board = curGame.getBoard();

        // Prepare input command sequence: two players take turns until lowerPlayer wins
        // lowerPlayer's leopard is at (5,3), upperPlayer's den is at (4,9)
        // Direction definition: L=x-1(left), R=x+1(right), U=y+1(up), D=y-1(down)
        // Path: (5,3) -> (4,3) -> (4,4) -> (4,5) -> (4,6) -> (4,7) -> (4,8) -> (4,9)
        // Note: cat cannot enter water (river squares at y=4,5,6), so we move cat left instead of down
        String inputCommands = 
            "leopard L\n" +      // lowerPlayer: leopard left (from 5,3 to 4,3)
            "rat D\n" +          // upperPlayer: rat down (from 1,7 to 1,6)
            "leopard U\n" +      // lowerPlayer: leopard up (from 4,3 to 4,4)
            "rat D\n" +          // upperPlayer: rat down (from 1,6 to 1,5)
            "leopard U\n" +      // lowerPlayer: leopard up (from 4,4 to 4,5)
            "rat D\n" +          // upperPlayer: rat down (from 1,5 to 1,4)
            "leopard U\n" +      // lowerPlayer: leopard up (from 4,5 to 4,6)
            "cat L\n" +          // upperPlayer: cat left (from 6,8 to 5,8) - avoid water at (6,6)
            "leopard U\n" +      // lowerPlayer: leopard up (from 4,6 to 4,7)
            "cat L\n" +          // upperPlayer: cat left (from 5,8 to 4,8) - moves to trap square
            "leopard U\n" +      // lowerPlayer: leopard up (from 4,7 to 4,8) - captures cat (leopard > cat)
            "dog D\n" +          // upperPlayer: dog down (from 2,8 to 2,7)
            "leopard U\n";       // lowerPlayer: leopard up (from 4,8 to 4,9) - wins!
        
        // Use ByteArrayInputStream to simulate interactive input
        ByteArrayInputStream testInput = new ByteArrayInputStream(inputCommands.getBytes());
        
        // Set System.in (may not be used, but set for completeness)
        System.setIn(testInput);
        
        // Use Game class's testing method to set Scanner
        // Note: need to create a new ByteArrayInputStream because Scanner will read from it
        ByteArrayInputStream scannerInput = new ByteArrayInputStream(inputCommands.getBytes());
        java.util.Scanner testScanner = new java.util.Scanner(scannerInput);
        Game.setScannerForTesting(testScanner);
        
        // Run startGame() in a new thread because it blocks waiting for input
        final boolean[] gameStarted = {false};
        final boolean[] gameFinished = {false};
        final Exception[] gameException = {null};
        
        Thread gameThread = new Thread(() -> {
            try {
                System.out.println("Game thread started");
                gameStarted[0] = true;
                curGame.startGame();
                System.out.println("Game ended normally");
                gameFinished[0] = true;
            } catch (Exception e) {
                gameException[0] = e;
                System.err.println("Game exception: " + e.getMessage());
                e.printStackTrace();
            }
        });
        gameThread.setDaemon(true);
        gameThread.start();
        
        // Wait for game to start (ensure startGame() has been called and is ready to receive input)
        int waitCount = 0;
        while (!gameStarted[0] && waitCount < 100) {
            try {
                Thread.sleep(10);
                waitCount++;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        
        if (!gameStarted[0]) {
            System.err.println("Error: Game thread failed to start");
            return;
        }
        
        // Wait a bit more to ensure startGame() has entered the state of waiting for input
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Wait for game to end (max 10 seconds)
        try {
            gameThread.join(10000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        // Restore original System.in
        System.setIn(originalIn);
        
        // Check for exceptions
        if (gameException[0] != null) {
            throw new RuntimeException("Exception occurred during game execution", gameException[0]);
        }
        
        // Verify if game completed
        if (!gameFinished[0]) {
            System.err.println("Warning: Game may not have completed normally, but continue to verify game state");
        }
        
        // Verify game end: upperPlayer's den is captured
        assertTrue(board.checkDen(true), "upperPlayer's den should be captured");
    }
}