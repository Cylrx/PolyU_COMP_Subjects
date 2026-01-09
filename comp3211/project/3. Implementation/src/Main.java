import java.util.Scanner;
import pages.game.Game;
import pages.loading.LoadGame;
import pages.loading.LoadReplay;

public class Main {
    private static final Scanner sc = new Scanner(System.in);
    private static void printWelcome() {
        System.out.println("Welcome to Jungle Game v1.0!");
    }
    public static void main(String[] args) {
        printWelcome();
        while(true) {
            System.out.flush();
            System.out.println();
            System.out.println("Main Menu");
            System.out.print("> ");
            String command = sc.nextLine();
            String[] arr = command.split("\\s+");
            if(arr.length == 0) {
                continue;
            }
            Game currentGame;
            if(arr.length > 1) {
                System.out.println("Too many arguments, please enter again");
                continue;
            }
            if(arr[0].equalsIgnoreCase("startNewGame")) {
                currentGame = Game.newGame();
                currentGame.startGame();
            } else if(arr[0].equalsIgnoreCase("loadGame")) {
                currentGame = LoadGame.mainLoop();
                if(currentGame != null) {
                    currentGame.startGame();
                }
            } else if(arr[0].equalsIgnoreCase("watchReplay")) {
                LoadReplay.mainLoop();
            } else if(arr[0].equalsIgnoreCase("exit")) {
                break;
            } else {
                System.out.println("Invalid command");
            }
        }
    }
}