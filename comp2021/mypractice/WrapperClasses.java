import static helper.PrintHelper.divider;
import static helper.PrintHelper.p;

import java.math.*;

public class WrapperClasses {
	public static void main(String[] args) {
		learnCharacter();
		learnStrings();
		learnBignum();
	}

	static void learnCharacter() {
		divider("Char Basics");

		char a = 'A';			// ASCII, A
		char b = (char)(a + 1);	// ASCII, B
		char four = '4';		// ASCII, 4
		char li = '\u7406';		// Unicode, 理
		char gong = '\u5DE5';	// Unicode, 工
		p("A: \t" + a);
		p("A+1: \t" + b);
		p("4: \t" + four);
		p("u7406: \t" + li);
		p("u5DE5: \t" + gong);

		divider("Char Operations");

		int i = 'a'; 		// 97
		char c = (char)i; 	// a
		p("int <- char (int = 'a'): " + i);
		p("char <- int (char = 97): " + c);
		
		divider("Character Class Methods");
		p("c == " + c);
		
		// These methods are static. So they can't be called on objects (e.g., c.isDigit())
		p("isDigit(c): " + Character.isDigit(c));
		p("isLetter(c): " + Character.isLetter(c));
		p("isLetterOrDigit(c): " + Character.isLetterOrDigit(c));
		p("isLowerCase(c): " + Character.isLowerCase(c));
		p("isUpperCase(c): " + Character.isUpperCase(c));
		p("toLowerCase(c): " + Character.toLowerCase(c));
		p("toUpperCase(c): " + Character.toUpperCase(c));

		divider("Autoboxing & Unboxing");
		
		Character CC = 'c';
		p("Boxing (char -> Character): " + CC);
		char cc = CC;
		p("Unboxing (Character -> char): " + cc);
	}

	static void learnStrings() {
		divider("String Basics");

		String s1 = "Hello ";
		String s2 = "Hel";
		p("String s1 = " + s1);
		p("String s2 = " + s2);

		divider("String Methods");
		p("s1.length():\t\t" + s1.length());
		p("s1.charAt(0):\t\t" + s1.charAt(0));
		p("s1.trim():\t\t" + s1.trim() + "|"); // Remove white space on both side;
		p("s1.substring(0,3):\t" + s1.substring(0, 3)); // [0, 3), not including 3;

		divider("String Methods (w/ two String)");
		p("s1.concat(s2):\t\t" + s1.concat(s2));
		p("s1.equals(s2):\t\t" + s1.equals(s2));
		p("s1.startsWith(s2):\t" + s1.startsWith(s2));
		p("s1.endsWith(s2):\t" + s1.endsWith(s2));

		// If NOT found, return -1
		p("s1.indexOf('l',0):\t" + s1.indexOf('l', 0)); // first position where 'l' appears, starting from 0
		p("s1.indexOf(s2, 0):\t" + s1.indexOf(s2, 0)); // first position where s2 appears, starting from 0
		p("s1.lastIndexOf('l',5):\t" + s1.lastIndexOf('l', 3));
		p("s1.lastIndexOf(s2, 5):\t" + s1.lastIndexOf(s2, 5));

		divider("String -> Numeric");
		p("Integer.parseInt(\"123\"):\t" + Integer.parseInt("123"));
		p("Double.parseDouble(\"123.456\"):\t" + Double.parseDouble("123.456"));
		p("+ Operator becomes string concatenation even if ONE operand is String");

		divider("StringBuilder");
		p("Motivation: Without StringBuilder, concatenating N Strings is O(N^2) Memory Complexity");
		p("StringBuilder comes with java.lang, thus NO NEED to import");
		
		StringBuilder sb = new StringBuilder("");
		for (int i = 97; i <= 107; i++) sb.append((char)i);
		p("Concat 'a'~'k':\t" + sb.toString());

	}

	static void learnBignum () {
		divider("BigInteger");
		BigInteger a = new BigInteger("9223372036854775807");
		BigInteger b = new BigInteger("1000000000000000000");
		p("BigInteger a: " + a);
		p("BigInteger b: " + b);
		p("a x b = " + a.multiply(b));
		p("a + b = " + a.add(b));

	}
}
