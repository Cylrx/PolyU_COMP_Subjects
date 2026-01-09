package pages.loading;

import pages.replay.Replay;

import java.io.*;
import java.util.*;

public class LoadReplay {
    private static final Scanner sc = new Scanner(System.in);
    private static final String replayPath = "./replay";
    
    public static void printLoadFiles() {
        File dir = new File(replayPath);
        if(!dir.exists() || !dir.isDirectory()) {
            System.out.println("No replay directory found.");
            return;
        }
        
        File[] files = dir.listFiles((d, name) -> name.endsWith(".record"));
        if(files == null || files.length == 0) {
            System.out.println("No saved replays found in replay directory.");
            return;
        }

        Arrays.sort(files, Comparator.comparing(File::getName));

        System.out.println("Available saved replays:");
        for(int i = 0; i < files.length; i++) {
            String filename = files[i].getName();
            String displayName = filename.substring(0, filename.length() - 7);
            System.out.println((i + 1) + ". " + displayName);
        }
    }
    
    public static void mainLoop() {
        while(true) {
            File dir = new File(replayPath);
            if(!dir.exists() || !dir.isDirectory()) {
                System.out.println("No replay directory found.");
                return;
            }

            File[] files = dir.listFiles((d, name) -> name.endsWith(".record"));
            if(files == null || files.length == 0) {
                System.out.println("No saved replays found in replay directory.");
                return;
            }

            List<File> fileList = new ArrayList<>(Arrays.asList(files));
            fileList.sort(Comparator.comparing(File::getName));

            printLoadFiles();
            System.out.println((fileList.size() + 1) + ". Return to main menu");
            System.out.print("Please select a replay to load or quit(enter number): ");
            
            String input = sc.nextLine().trim();
            
            try {
                int choice = Integer.parseInt(input);
                
                if(choice == fileList.size() + 1) {
                    return;
                } else if(choice >= 1 && choice <= fileList.size()) {
                    File selectedFile = fileList.get(choice - 1);
                    try {
                        FileInputStream fis = new FileInputStream(selectedFile);
                        ObjectInputStream ois = new ObjectInputStream(fis);
                        Replay loadedReplay = (Replay) ois.readObject();
                        ois.close();
                        fis.close();
                        System.out.println("Replay loaded successfully: " + selectedFile.getName());
                        loadedReplay.startReplay();
                        System.out.println("Replay finished. Returning to replay selection...");
                    } catch (IOException e) {
                        System.out.println("Error loading replay: " + e.getMessage());
                        System.out.println("Please try again.");
                    } catch (ClassNotFoundException e) {
                        System.out.println("Error: Replay file format is invalid.");
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

