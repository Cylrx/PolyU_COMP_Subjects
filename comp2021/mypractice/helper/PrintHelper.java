package helper;

public class PrintHelper {
	public static void divider (String name) {
		System.out.println("\n--- " + name + " ---");
	}

	// Shorthand for println
	public static void p(Object... args) {
  		System.out.println(args[0]);
	}
}
