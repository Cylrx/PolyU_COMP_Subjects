import static helper.PrintHelper.p;
import static helper.PrintHelper.divider;

public class AdvancedSyntax {
	public static void main (String[] args) {
		learnScope();
	}

	private static void learnScope () {
		divider("Java Scopes");
		
		int x = 10;
		if (x == 10) {
			int x = 20;
		

	}
}
