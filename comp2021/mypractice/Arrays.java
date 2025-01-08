import static helper.PrintHelper.p;
import static helper.PrintHelper.divider;

public class Arrays {
	public static void main (String[] args) {
		array1D();
		arrayND();
	}

	static void array1D () {
		divider("1D Array");

		double[] myDoubles = new double[10];
		double[] initDoubles = {1.0, 2.2, 3.4};

		p("myDoubles.length:\t" + myDoubles.length);
		p("initDoubles.length:\t" + initDoubles.length);

		myDoubles = new double[]{1, 2, 3};
		p("after reassign:\t\t" + myDoubles.length);
	}

	static void arrayND () {
		divider("2D Array");
		double[][] myDoubles = new double[2][4];
		p("new double[2][4]'s length:\t" + myDoubles.length);

		divider("Ragged Array");
		int[][] myInteger = {
			{1, 2, 3, 4, 5},
			{1, 2, 3, 4},
			{1, 2, 3},
			{1, 2},
			{1}
		};
		// Or could be written as:
		// int[][] a = new int[5][];
		// a[0] = new int[5];
		// a[1] = new int[4];
		// a[2] = new int[3];
		// a[3] = new int[2];
		// a[4] = new int[1];

		for (int i = 0; i < 5; i++) p("length of myInteger[" + i + "]\t" + myInteger[i].length);

		divider("3D Ragged Array");
		int[][][] arr = new int[3][][];
		arr[0] = myInteger;
		arr[1] = myInteger;
		arr[2] = myInteger;
		
		for (int i = 0; i < 3; i++) {
			p("__a[" + i + "][][]__");
			for (int j = 0; j < 5; j++) {
				p("a[" + i + "]" + "[" + j + "][].length = " + arr[i][j].length);
			}
		}
	}
}
