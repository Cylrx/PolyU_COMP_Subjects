import java.util.Scanner;

public class PostfixNotation {
	public static boolean isNum (char x) {
		return '0' <= x && x <= '9';
	}

	private static int getVal (char x) {
		if (x == '*' || x == '/') return 2;
		if (x == '+' || x == '-') return 1;
		return 0;
	}
	
	public static boolean isBetter (char a, char b) {
		return getVal(a) > getVal(b);
	}

	public static void convert (char[] a) {
		int n = a.length;
		char[] res = new char[n];
		char[] sign = new char[n];
		int j = 0, sp = -1;
		for (int i = 0; i < n; i++) {
			if (isNum(a[i])) {
				res[j++] = a[i];
			} else {
				if (a[i] == '(') sign[++sp] = a[i];
				else if (a[i] == ')') {
					while (sign[sp] != '(') {
						res[j++] = sign[sp];
						sp--;
					}
					sp--;
				} else {
					while (sp != -1 && !isBetter(a[i], sign[sp])) {
						res[j++] = sign[sp];
						sp--;
					}
					sign[++sp] = a[i];
				}
			}
		}
		while (sp >= 0) {
			res[j++] = sign[sp];
			sp--;
		}
		for (int i = 0; i < n; i++) {
			a[i] = res[i];
		}
	}

	public static void main (String[] args) {
		Scanner scan = new Scanner(System.in);		
		String str = scan.nextLine();
		char[] exp = str.toCharArray();
		convert(exp);
		for (int i = 0; i < exp.length; i++) {
			System.out.print(exp[i]);
		}
	}
}
