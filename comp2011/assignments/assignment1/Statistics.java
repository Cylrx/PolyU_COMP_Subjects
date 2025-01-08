package assignment1;

/**
 * @author Yixin Cao (September 11, 2024)
 * Calculates the median of one or more sorted arrays.
 */

public class Statistics {
    /**
     * Calculates the median of a <i>sorted</i> array.
     * The input is assumed to be <i>nonempty</i> and <i>sorted</i>.
     * 
     * BTW, this example may help you understand the ugliness behind the light word "evenly."
     *
     * @throws IllegalArgumentException if the input array is empty
     */

    public static double median(int[] a) {
        int n = a.length;
        if (n == 0) throw new IllegalArgumentException("The input must be nonempty.");
        int mid = n / 2;
        if (mid * 2 != n) return a[mid];
        return (a[mid] + a[mid - 1]) / 2.0; 
    }
    
    /**
     * Calculates the median of the union of two <i>sorted</i> arrays.
     * The input arrays are both <i>sorted</i> and <i>cannot</i> be both empty.
     *
     * VERY IMPORTANT.
     * 
     * I've discussed this question with the following students:
     *     1. Yang Xikun
	 *     2. Yang Jinkun
     * 
     * I've sought help from the following Internet resources and books:
     *     1. None
     * 
     * Running time: O(log(n)); space O(1).   
     */

    public static double median(int[] a1, int[] a2) {
		int n1 = a1.length;
		int n2 = a2.length;
		if (n1 == 0) return median(a2);
		if (n2 == 0) return median(a1);
		
		if ((n1 + n2) % 2 == 1) { 
			int k = ((n1 + n2) >> 1) + 1;
			return findKth(a1, a2, k);
		} else {
			int k1 = ((n1 + n2) >> 1) + 1;
			int k2 = (n1 + n2) >> 1;
			return (findKth(a1, a2, k1) + findKth(a1, a2, k2)) / 2.0;
		}
    }

	/**
	 * Calculates the k'th element of a single array.
	 * Preconditions
	 *  - a.length cannot be less than k
	 *  - a cannot be empty
	 *
	 * Running time: O(1); Space: O(1).
	 *
	 * @param a the array in question
	 * @param k the k'th element
	 * @return the value of the k'th element
	 * @throws IllegalArgumentException if any of the above preconditions are unmet
	 */
	private static int findKth(int[] a, int k) {
		int n = a.length;
		if (n == 0) throw new IllegalArgumentException("The input must be nonempty");
		if (n < k) throw new IllegalArgumentException("K must be less than or equal to the length of a");
		return a[k-1];
	}

	/**
	 * Calculates the k'th element of the sorted union of two arrays
	 * Preconditions
	 *  - a1 and a2 cannot be both empty at the same time
	 *  - a1.length + a2.length cannot be less than k
	 *
	 * Main Logic:
	 *  - (Assume array indices start at 1)
	 *  - To find k-th element, compare a1[k1] and a2[k2] where k1 + k2 = k
	 *  - It can be proven that, the array with the smaller value at k1/k2 cannot contain the k-th element
	 *  - For example, if a[k1] less than a[k2], then a[0...k1] excludes  k'th element
	 *  - By selecting k1 and k2 ≈ k/2 each step, the algorithm achieves O(log(k)) time complexity.
	 *
	 *  Running time: O(log(k)); Space: O(1);
	 *
	 * @param a1 the first array
	 * @param a2 the second array
	 * @param k the k'th element
	 * @return the value of the k'th element
	 * @throws IllegalArgumentException if any of the above preconditions are unmet
	 */
	private static int findKth(int[] a1, int[] a2, int k) {
		int n1 = a1.length;
		int n2 = a2.length;
		if (n1 == 0 && n2 == 0) throw new IllegalArgumentException("The input arrays cannot be both empty");
		if (n1 + n2 < k) throw new IllegalArgumentException("total length of the input arrays cannot be less than k");

		int i1 = 0, i2 = 0;
		while (n1 > 0 && n2 > 0) {
			if (k == 1) break;
			int k1 = Math.min(n1, (k >> 1));
			int k2 = Math.min(n2, (k - k1));
			if (a1[i1 + k1 - 1] < a2[i2 + k2 - 1]) {
				i1 += k1; 
				n1 -= k1;
				k -= k1;
			} else {
				i2 += k2;
				n2 -= k2;
				k -= k2;
			}
		}
		if (n1 == 0) return findKth(a2, k + i2);
		if (n2 == 0) return findKth(a1, k + i1);
		return Math.min(a1[i1], a2[i2]);
	}

    public static void main(String[] args) {
        int[][] a = {{-2, 0, 1, 3}, {1, 2, 4, 8, 16}};
        System.out.println("The medians of the two arrays are " + median(a[0]) + ", " + median(a[1])
        + ", and the median of their union is " + median(a[0], a[1]));
	}
}
