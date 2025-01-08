package hk.edu.polyu.comp.comp2021.cvfs;

import hk.edu.polyu.comp.comp2021.cvfs.view.View;
import hk.edu.polyu.comp.comp2021.cvfs.controller.Controller;

import java.util.Scanner;

/**
 * This is the main class of the CVFS.
 * The CVFS is a virtual file system that allows users to create, delete, and manage directories and documents.
 * It allows the creation of criteria to filter files and directories.
 * Recursive or non-recursive search is hence supported with or without the criterias.
 */

public class Application {
    /**
     * The entry point of the CVFS.
     * The CVFS will prompt the user for input and execute the corresponding command.
     * It exits when the user types "quit".
     * @param args the command line arguments
     */
    public static void main(String[] args){
        View view = new View();
        Controller control = new Controller(view);
        Scanner scan = new Scanner(System.in);
        boolean isRunning = true;

        while (isRunning) {
            String input = scan.nextLine().trim();
            isRunning = control.runInstruction(input);
        }
        scan.close();
        view.displayMessage("CVFS has exited gracefully.");
    }
}
