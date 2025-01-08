package utils;

import view.View;

import java.io.Console;

import static controller.Controller.sc;

public class InputUtils {
    /**
     * Prompts the user with <code>message</code> and returns the String input as a string.
     * @param message the message to display to the user
     * @return the input string entered by the user
     */
    public static String getStr(String message) {
        View.displayPrompt(message);
        return sc.nextLine();
    }

    public static int getDigit(String message) {
        while (true) {
            View.displayPrompt(message);
            String strOp = sc.nextLine().trim();
            if (strOp.length() != 1 || !Character.isDigit(strOp.charAt(0))) {
                View.displayBadInput("Single digit", strOp);
                continue;
            }
            return Integer.parseInt(strOp);
        }
    }

    public static char[] getPasswd(String message) {
        View.displayPrompt(message);
        Console console = System.console();
        return console.readPassword();
    }

    public static char[] getNewPasswd(String message){
        while (true) {
            char[] input = getPasswd(message);
            if (SecurityUtils.eval(input)) return input;
            else View.displayError(SecurityUtils.reason());
        }
    }
}
