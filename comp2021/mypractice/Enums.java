import static helper.PrintHelper.p;
import static helper.PrintHelper.divider;

public class Enums {
	public static void main (String[] args) {
		enumsDemo();
	}

	// Simple enum
	public enum Suit {
		CLUBS, DIAMONDS, HEARTS, SPADES;
	}

	// Complex enum
	public enum Daytime {
		MORNING(7),
		NOON(12),
		AFTERNOON(16),
		EVENING(18),
		MIDNIGHT(24);
		
		private int time;
		Daytime (int time) { this.time = time;}
		public int getTime () { return time; }

		public String to24HFormat () {
			int f = (this.time - 1) % 12 + 1;
			if (this.time < 12) return f + "AM";
			return f + "PM";
		}

	}

	static void enumsDemo () {
		divider("Simple Enum (Suit)");
		Suit s = Suit.CLUBS;
		p("Suit.CLUBS = " + s);
		p("\ncompareTo compares by order of declaration");
		p("s.compareTo(Suit.CLUBS)  = " + s.compareTo(Suit.CLUBS));
		p("s.compareTo(Suit.HEARTS) = " + s.compareTo(Suit.DIAMONDS));
		p("s.compareTo(Suit.HEARTS) = " + s.compareTo(Suit.HEARTS));

		p("\ns == Suit.CLUBS: " + (s == Suit.CLUBS));
		
		divider("Complex Enum (Daytime)");

		Daytime d = Daytime.MORNING;
		p("d = " + String.valueOf(d));
		p("d.getTime() = " + d.getTime());
		p("d.to24HFormat() = " + d.to24HFormat());

		Daytime n = Daytime.MIDNIGHT;
		p("\nn = " + String.valueOf(n));
		p("d.getTime() = " + d.getTime());
		p("d.to24HFormat() = " + d.to24HFormat());

		p("\ncomparing");
		p("d.compareTo(n) = " + d.compareTo(n) + " (Still compares based on order of declaration, NOT value)"); 
	}
}
