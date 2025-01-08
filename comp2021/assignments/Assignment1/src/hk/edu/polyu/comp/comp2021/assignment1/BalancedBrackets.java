package hk.edu.polyu.comp.comp2021.assignment1;


public class BalancedBrackets {

    public static boolean isBalanced(String str){
        // Task 7: Return true if and only if 'str' 1) is non-empty,
        // 2) contains only the six characters, and
        // 3) is balanced.

        char[] s = str.toCharArray();
        int n = s.length;
        if (n == 0) return false;

        char[] stack = new char[n];
        int sp = 0;
        for (char c: s) {
            if (c == '(' || c == '[' || c == '{') { stack[sp++] = c; continue; }
            if (c == ')' || c == ']' || c == '}') {
                if (sp == 0) return false;
                char t = stack[sp - 1];
                if ((t == '(' && c == ')') || (t == '[' && c == ']') || (t == '{' && c == '}')) sp--;
                else return false;
            }
            else return false;
        }
        return sp == 0;
    }
}
