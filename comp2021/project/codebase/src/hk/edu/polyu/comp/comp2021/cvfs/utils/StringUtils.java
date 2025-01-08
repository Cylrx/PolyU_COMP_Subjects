package hk.edu.polyu.comp.comp2021.cvfs.utils;

import hk.edu.polyu.comp.comp2021.cvfs.exceptions.InvalidCommandException;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Utility class for string manipulation.
 */
public class StringUtils {

    /**
     * Split a string by spaces, <b>except</b> when the space is inside a pair of backticks.
     * @param strInp the input string
     * @return an array of strings
     * @throws InvalidCommandException when the backticks are not properly used (i.e., not surrounded by ' ' or not closed)
     */
    public static String[] splitStr(String strInp) throws InvalidCommandException {
        ArrayList<String> tmp = new ArrayList<>();
        StringBuilder sb = new StringBuilder();

        int n = strInp.length();
        char[] inp = strInp.toCharArray();
        boolean inQuote = false;

        for (int i = 0; i < n; i++) {
            if (inp[i] == '`') {
                if ((!inQuote && sb.length() != 0) || (inQuote && i != n-1 && inp[i+1] != ' '))
                    throw new InvalidCommandException("backtick (`) must be surrounded by ' ' or nothing");
                inQuote = !inQuote;
                continue;
            }
            if (inp[i] == ' ' && !inQuote) {
                tmp.add(sb.toString());
                sb.setLength(0);
            } else sb.append(inp[i]);
        }
        if (inQuote) throw new InvalidCommandException("backtick (`) not closed");
        if (sb.length() > 0) tmp.add(sb.toString());

        n = tmp.size();
        String[] ans = new String[n];
        for (int i = 0; i < n; i++) ans[i] = tmp.get(i);
        return ans;
    }

    /**
     * Repeat a character for a given number of times.
     * @param sub the character to be repeated
     * @param cnt the number of times to repeat
     * @return the repeated string
     */
    public static String repeatStr(char sub, int cnt) {
        char[] res = new char[cnt];
        Arrays.fill(res, sub);
        return new String(res);
    }
}
