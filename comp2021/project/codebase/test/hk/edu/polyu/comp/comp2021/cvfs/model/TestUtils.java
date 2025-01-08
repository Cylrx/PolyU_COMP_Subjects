package hk.edu.polyu.comp.comp2021.cvfs.model;

import java.util.Random;

public class TestUtils {
    private static final char[] CHAR_ARRAY = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".toCharArray();
    private static final int CHAR_LENGTH = CHAR_ARRAY.length;

    public static String repeat(String base, int cnt) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < cnt; i++) sb.append(base);
        return sb.toString();
    }

    public static String getRandFileType(Random rand){
        String[] types = {"txt","java","html","css"};
        return types[rand.nextInt(types.length)];
    }

    public static String getRandStr(Random rand, int length){
        char[] result = new char[length];
        for (int i = 0; i < length; i++) result[i] = CHAR_ARRAY[rand.nextInt(CHAR_LENGTH)];
        return new String(result);
    }

    public static boolean notValidFileType(String type){
        return !type.equals("txt") && !type.equals("java") && !type.equals("html") && !type.equals("css");
    }

    public static boolean compare(long a, String operator, long b){
        switch(operator){
            case "<": return a < b;
            case ">": return a > b;
            case "<=": return a <= b;
            case ">=": return a >= b;
            case "==": return a == b;
            case "!=": return a != b;
            default: return false;
        }
    }
}
