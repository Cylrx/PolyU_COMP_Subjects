package pages.game;

import pages.display.Display;
import model.Board;
import pages.replay.Replay;

import java.io.*;
import java.time.LocalDateTime;
import java.util.Scanner;

public class Game implements Serializable {
    private final Board board;
    private final Player upperPlayer, lowerPlayer;
    private boolean currentTurn = false; // false means lowerPlayer
    private boolean isActive;

    private static Scanner scanner = new Scanner(System.in);
    private static final String replayPath = "./replay";
    private static final String archivePath = "./archive";

    // For testing only
    public Game() {
        board = new Board();
        upperPlayer = new Player(null);
        lowerPlayer = new Player(null);
        isActive = true;
    }
    
    /**
     * Create a new game with the given player names.
     *
     * @param firstName  name for the lower player; if null, Player will typically
     *                   generate a random name.
     * @param secondName name for the upper player; if null, Player will typically
     *                   generate a random name.
     */
    private Game(String firstName, String secondName) {
        board = new Board();
        lowerPlayer = new Player(firstName);
        upperPlayer = new Player(secondName);
        isActive = true;
    }
    
    /**
     * Save the current game state into a {@code .jungle} archive file.
     * <p>
     * This uses Java serialization to write the entire {@link Game} object
     * (including board and player state) into {@code ./archive/<filename>.jungle}.
     * The directory {@code ./archive} will be created if it does not exist.
     *
     * @param filename base filename without extension; must match {@code [a-zA-Z0-9. _-]+}.
     * @throws IllegalArgumentException if the filename is illegal or the file cannot be written.
     */
    public void saveGame(String filename) {
        File dir = new File(archivePath);
        if(!dir.exists()) {
            dir.mkdirs();
        }
        if(!filename.matches("^[a-zA-Z0-9. _-]+$")) {
            throw new IllegalArgumentException("Illegal filename format");
        }
        try {
            FileOutputStream fos = new FileOutputStream("./archive/" + filename + ".jungle");
            ObjectOutputStream out = new ObjectOutputStream(fos);
            System.out.println("Saving game to " + "./archive/" + filename + ".jungle");
            out.writeObject(this);
            out.close();
            fos.close();
        } catch (IOException e) {
            System.err.println(e.getMessage());
            throw new IllegalArgumentException("File may already exist but can not modify!");
        }
        System.out.println("Save successful!");
    }
    
    /**
     * Mark this game as inactive.
     * <p>
     * The main input loop in {@link #startGame()} will exit once {@code isActive}
     * becomes {@code false}. This does not perform any persistence by itself.
     */
    public void closeGame() {
        isActive = false;
    }
    
    /**
     * Save a replay snapshot of the current board and player names into a {@code .record} file.
     * <p>
     * Note that this does NOT re-serialize the full {@code Game} object, but rather
     * creates a {@link Replay} instance from the current {@link Board} and both player names
     * and writes that object into {@code ./replay/<filename>.record}.
     *
     * @param filename base filename without extension; must match {@code [a-zA-Z0-9. _-]+}.
     * @throws IllegalArgumentException if the filename is illegal or the file cannot be written.
     */
    public void saveReplay(String filename) {
        File dir = new File(replayPath);
        if(!dir.exists()) {
            dir.mkdirs();
        }
        if(!filename.matches("^[a-zA-Z0-9. _-]+$")) {
            throw new IllegalArgumentException("Illegal filename format");
        }
        Replay curReplay = new Replay(board, lowerPlayer.getName(), upperPlayer.getName());
        try {
            FileOutputStream fos = new FileOutputStream("./replay/" + filename + ".record");
            ObjectOutputStream out = new ObjectOutputStream(fos);
            System.out.println("Saving replay to" + "./replay/" + filename + ".record");
            out.writeObject(curReplay);
            out.close();
            fos.close();
        } catch (IOException e) {
            System.err.println(e.getMessage());
            throw new IllegalArgumentException("File may already exist but can not modify!");
        }
        System.out.println("Save successful!");
    }
    
    /**
     * Start the interactive command-line game loop.
     * <p>
     * This method:
     * <ul>
     *   <li>Marks the game as active.</li>
     *   <li>Displays the initial board.</li>
     *   <li>Repeatedly prompts the current player for a command until the game ends.</li>
     * </ul>
     * The supported commands are:
     * <ul>
     *   <li>{@code &lt;pieceName&gt; &lt;direction&gt;} – move a piece (e.g. {@code rat U}).</li>
     *   <li>{@code withdraw} – undo the last move for the current player (quota-limited).</li>
     *   <li>{@code saveGame [filename]} – save current game as {@code .jungle}.</li>
     *   <li>{@code saveReplay [filename]} – save current replay snapshot as {@code .record}.</li>
     *   <li>{@code close} – exit the game loop without automatic saving.</li>
     * </ul>
     * Input lines are parsed by simple whitespace splitting; invalid commands will print
     * an error message and prompt the player again.
     */
    public void startGame() {
        isActive = true;
        Display.displayBoardWithLegend(board);
        while(isActive) {
            // Before each turn, check whether the game should end:
            //  - checkDen(currentTurn)  → someone has reached the opponent's den.
            //  - !haveValidMove(...)    → current player has no legal moves left.
            if(board.checkDen(currentTurn) || !board.haveValidMove(currentTurn)) {
                gameOver(!currentTurn);
                continue;
            }
            System.out.flush();
            System.out.println();
            System.out.print(getFullName(currentTurn) + " > ");

            // Read one line of user input, then split by whitespace into tokens.
            String line = scanner.nextLine();
            String[] arr = line.split("\\s+");

            // Empty line: ignore and prompt again.
            if(arr.length == 0) {
                continue;
            }

            // Branch 1: first token looks like a piece name → treat this as a move command.
            if(Utils.checkPieceName(arr[0])) {
                // Expect exactly 2 tokens: "<pieceName> <direction>".
                if(arr.length > 2) {
                    System.out.println("Too many argument, please enter again");
                    continue;
                }
                if(arr.length < 2) {
                    System.out.println("Please indicate a direction, try again");
                    continue;
                }

                // Validate direction format (e.g. U/D/L/R).
                if(!Utils.checkMove(arr[1])) {
                    System.out.println("Illegal move, please enter again");
                    continue;
                }
                // Delegate the actual move logic to Board.attemptMove.
                // If the move succeeds, toggle the turn and display the updated board.
                if(board.attemptMove(arr[0].toLowerCase(), currentTurn, Character.toUpperCase(arr[1].charAt(0)))) {
                    currentTurn = !currentTurn;
                    Display.displayBoard(board);
                }

            // Branch 2: "withdraw" command – undo the last move for the current player.
            } else if(arr[0].equalsIgnoreCase("withdraw")) {
                // This command should not have any extra arguments.
                if(arr.length > 1) {
                    System.out.println("Too many argument, please enter again");
                }
                // Check whether the current player still has withdraw quota left.
                if(currentTurn ? upperPlayer.checkWithdrawQuota() : lowerPlayer.checkWithdrawQuota()) {
                    try {
                        // Ask the board to undo the last move for this player.
                        board.tryWithdraw(currentTurn);
                        // Deduct withdraw quota for the current player.
                        if (currentTurn) {
                            upperPlayer.deductWithdrawQuota();
                        } else {
                            lowerPlayer.deductWithdrawQuota();
                        }
                        System.out.println("Withdraw successful! You have " + (currentTurn ? upperPlayer.getWithdrawQuota() : lowerPlayer.getWithdrawQuota()) + " withdraw Quota left");
                        Display.displayBoard(board);
                    } catch (IllegalArgumentException e) {
                        // Board throws when there is no move to withdraw.
                        System.out.println("You have not made any move!");
                    }
                } else {
                    System.out.println("You do not have enough withdraw quota!");
                }

            // Branch 3: "saveGame" command – serialize the whole Game into a .jungle file.
            } else if(arr[0].equalsIgnoreCase("saveGame")) {
                if(arr.length > 2) {
                    System.out.println("Too many argument, please enter again");
                } else if(arr.length > 1) {
                    try {
                        saveGame(arr[1]);
                    } catch (Exception e) {
                        System.out.println(e.getMessage());
                    }
                } else {
                    try {
                        // No filename provided → use timestamp as filename.
                        saveGame(String.valueOf(LocalDateTime.now()).replace(':', '-'));
                    } catch (Exception e) {
                        System.out.println(e.getMessage());
                    }
                }

            // Branch 4: "saveReplay" command – save a lightweight replay snapshot.
            } else if(arr[0].equalsIgnoreCase("saveReplay")) {
                if(arr.length > 2) {
                    System.out.println("Too many argument, please enter again");
                } else if(arr.length > 1) {
                    try {
                        saveReplay(arr[1]);
                    } catch (Exception e) {
                        System.out.println(e.getMessage());
                    }
                } else {
                    try {
                        // No filename provided → use timestamp as filename.
                        saveReplay(String.valueOf(LocalDateTime.now()).replace(':', '-'));
                    } catch (Exception e) {
                        System.out.println(e.getMessage());
                    }
                }

            // Branch 5: "close" command – exit the loop gracefully.
            } else if(arr[0].equalsIgnoreCase("close")) {
                closeGame();

            // Fallback: unknown command.
            } else {
                System.out.println("Unknown command, please enter again");
            }
        }
    }
    
    /**
     * Build a human-readable label for the player whose turn it is.
     * <p>
     * Example: {@code "upperPlayer (Alice)"} or {@code "lowerPlayer (Bob)"}.
     *
     * @param turn {@code true} for upperPlayer, {@code false} for lowerPlayer.
     * @return formatted player label with side and name.
     */
    private String getFullName(boolean turn) {
        return turn ? "upperPlayer (" + upperPlayer.getName() + ")" : "lowerPlayer (" + lowerPlayer.getName() + ")";
    }
    
    /**
     * Handle end-of-game logic when a winner is determined.
     * <p>
     * This prints out the winner, marks the game as inactive so that
     * {@link #startGame()} will exit, and automatically saves a replay
     * with a timestamp-based filename.
     *
     * @param winner {@code true} if the upper player wins, {@code false} if the lower player wins.
     */
    private void gameOver(boolean winner) {
        System.out.println(getFullName(winner) + " wins!");
        System.out.println("Game over!");
        isActive = false;
        saveReplay(String.valueOf(LocalDateTime.now()).replace(':', '-'));
    }
    
    /**
     * Factory method for starting a new interactive game.
     * <p>
     * This method:
     * <ul>
     *   <li>Prompts for lower player's name (empty for random name).</li>
     *   <li>Prompts for upper player's name (empty for random name).</li>
     *   <li>Constructs a new {@link Game} with those names.</li>
     * </ul>
     *
     * @return a freshly initialized {@code Game} ready for {@link #startGame()}.
     */
    public static Game newGame() {
        System.out.print("Please enter the name for the lowerPlayer (empty for a random name) :> ");
        String name1 = scanner.nextLine();
        if(name1.isEmpty()) name1 = null;
        System.out.print("Please enter the name for the upperPlayer (empty for a random name) :> ");
        String name2 = scanner.nextLine();
        if(name2.isEmpty()) name2 = null;
        return new Game(name1, name2);
    }
    
    /**
     * Get the underlying board instance for this game.
     * <p>
     * Mostly used by tests or higher-level components that need direct access
     * to the model layer. Regular gameplay should go through {@link #startGame()}.
     *
     * @return the {@link Board} object associated with this game.
     */
    public Board getBoard() {
        return board;
    }

    // for testing only
    /**
     * Replace the static {@link Scanner} used for reading user input.
     * <p>
     * This is primarily for testing, so that unit tests can inject a scanner
     * that reads from a predefined input source instead of {@code System.in}.
     *
     * @param testScanner scanner instance to use instead of the default one.
     */
    static void setScannerForTesting(Scanner testScanner) {
        scanner = testScanner;
    }
}
