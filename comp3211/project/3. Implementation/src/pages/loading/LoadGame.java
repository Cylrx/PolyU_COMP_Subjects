package pages.loading;

import pages.game.Game;

import java.io.*;
import java.util.*;

public class LoadGame {
    private static final Scanner sc = new Scanner(System.in);
    private static final String archivePath = "./archive";
    
    public static void printLoadFiles() {
        File dir = new File(archivePath);
        if(!dir.exists() || !dir.isDirectory()) {
            System.out.println("No archive directory found.");
            return;
        }
        
        File[] files = dir.listFiles((d, name) -> name.endsWith(".jungle"));
        if(files == null || files.length == 0) {
            System.out.println("No saved games found in archive.");
            return;
        }

        Arrays.sort(files, Comparator.comparing(File::getName));

        System.out.println("Available saved games:");
        for(int i = 0; i < files.length; i++) {
            String filename = files[i].getName();
            String displayName = filename.substring(0, filename.length() - 7);
            System.out.println((i + 1) + ". " + displayName);
        }
    }
    
    public static Game mainLoop() {
        File dir = new File(archivePath);
        if(!dir.exists() || !dir.isDirectory()) {
            System.out.println("No archive directory found.");
            return null;
        }

        File[] files = dir.listFiles((d, name) -> name.endsWith(".jungle"));
        if(files == null || files.length == 0) {
            System.out.println("No saved games found in archive.");
            return null;
        }

        List<File> fileList = new ArrayList<>(Arrays.asList(files));

        printLoadFiles();
        System.out.println((fileList.size() + 1) + ". Return to main menu");
        while(true) {
            System.out.print("Please select a game to load or quit(enter number): ");
            
            String input = sc.nextLine().trim();
            
            try {
                int choice = Integer.parseInt(input);
                
                if(choice == fileList.size() + 1) {
                    return null;
                } else if(choice >= 1 && choice <= fileList.size()) {
                    File selectedFile = fileList.get(choice - 1);
                    try {
                        FileInputStream fis = new FileInputStream(selectedFile);
                        ObjectInputStream ois = new ObjectInputStream(fis);
                        Game loadedGame = (Game) ois.readObject();
                        ois.close();
                        fis.close();
                        System.out.println("Game loaded successfully: " + selectedFile.getName());
                        return loadedGame;
                    } catch (IOException e) {
                        System.out.println("Error loading game: " + e.getMessage());
                        System.out.println("Please try again.");
                    } catch (ClassNotFoundException e) {
                        System.out.println("Error: Game file format is invalid.");
                        System.out.println("Please try again.");
                    }
                } else {
                    System.out.println("Invalid number. Please enter a number between 1 and " + (fileList.size() + 1));
                }
            } catch (NumberFormatException e) {
                System.out.println("Invalid input. Please enter a number.");
            }
        }
    }
}
