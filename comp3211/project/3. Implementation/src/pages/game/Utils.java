package pages.game;

import java.util.Random;

public class Utils {
    private static final String validNameChar = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    private static final Random generator = new Random();

    private static final String[] validPiece = {"rat", "cat", "dog", "wolf", "leopard", "tiger", "lion", "elephant"};
    private static final String[] validMove = {"U", "D", "L", "R"};

    public static String getRandomString(int len) {
        StringBuilder ret = new StringBuilder();
        for(int i = 0; i < len; i++) {
            ret.append(validNameChar.charAt(generator.nextInt(validNameChar.length())));
        }
        return ret.toString();
    }

    public static boolean checkPieceName(String name) {
        for(String names : validPiece) {
            if(name.equalsIgnoreCase(names)) {
                return true;
            }
        }
        return false;
    }

    public static boolean checkMove(String move) {
        for(String names : validMove) {
            if(names.equalsIgnoreCase(move)) {
                return true;
            }
        }
        return false;
    }
}
