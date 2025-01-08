package assignment1;

import java.util.Arrays;

/**
 * 
 * @author Yixin Cao (September 11, 2021)
 *
 * A better implementation of insertion sort.
 * 
 * The ith major iteration of insertion sort puts element a[i] to its correct position.
 * Your algorithm should use <i>binary search</i> to find the position of the first element 
 * that is greater than (> not >=) a[i], or i if there does exist such an element.
 * 
 * The comparison is conducted via the {@link Comparable#compareTo(java.lang.Object)} method.  
 * Please be reminded that the comparisons are <i>not</i> on the values of the elements.
 * 
 * To facilitate your testing, we have included an extra field {@code originalPos} 
 * in the {@code Element} class. It stores the index of this element in the input array.
 * When an element is output, originalPos is printed in parentheses.
 * 
 * If your implementation is correct, elements of the same value should respect their original order,
 * e.g., for input {3, 10, 20, 3}, the output should be [10 (1), 3 (0), 20 (2), 3 (3)].
 */
public class Sorting { // Please change!
    /*
     * Each element has a secret and the original position in the input array.
     * Modifications of this class are strictly forbidden. 
     */
    public static class Element implements Comparable<Element> {
        private int originalPos;
        private int secret;
        
        public Element(int i, int s) {
            if (s <= 0) {
				System.out.println("Got invalid 's': " + s);
                throw new IllegalArgumentException("The secret value must be a positive integer."); 
            }
            originalPos = i; 
            secret = s;
        }
        
        /*
         * For this assignment, you do <i>not</i> need to understand the {@code compareTo} method.
         */
        @Override public int compareTo(Element other) {
            int a = secret, b = other.secret;
            if (a == b) return 0; // can be removed.
            while (a != 1 && b != 1) {
                if (a / 2 * 2 == a) a /= 2;
                else a = a * 3 + 1;
                if (b / 2 * 2 == b) b /= 2;
                else b = b * 3 + 1;
            }
            return a - b;
        }
        
        public String toString() {
            return (String.valueOf(secret)) + " (" + String.valueOf(originalPos) + ")";
        }
    }

	/**
     * Sorts an array of `Element` objects using an optimized insertion sort that reduces the number of `compareTo` calls required.
     *
     * VERY IMPORTANT.
     *
     * I've discussed this question with the following students:
     *     1. None.
     *
     * I've sought help from the following Internet resources and books:
     *     1. <a href="https://en.wikipedia.org/wiki/Collatz_conjecture">Collatz Conjecture (Wikipedia)</a>
     *
     * The algorithm works as follows:
     *  - A binary search is performed to find the correct position for the current element `x` in the sorted portion of the array.
     *  - The search maintains the <b>loop invariant</b> that `l` and `r` enclose the <b>first element</b> greater than `x`.
     *  - Once found, elements to its right (including itself) are shifted one place rightwards to make space for `x`.
     *  - Finally, `x` is inserted at the determined position.
     *
     * Time Complexity: O(n^2)
     * More precisely: O(n^2 + nlogn * T(m))
     * Where,
     *  - n is the number of elements in the array
     *  - m is the initial value of the collatz series
     *  - T(m) = O(logm * loglogm), empirically
     *
     * @param a the array of `Element` objects to be sorted
     */
    public static void insertionSort(Element[] a) {
        int n = a.length;
		for (int i = 1; i < n; i++) {
			Element x = a[i];
			int l = 0, r = i - 1;
			while (l < r) {
				int mid = (l + r) >> 1;
				if (a[mid].compareTo(x) > 0) r = mid;
				else l = mid + 1;
			}
			if (a[l].compareTo(x) > 0) {
				for (int j = i - 1; j >= l; j--) a[j + 1] = a[j];
				a[l] = x;
			}
		}
    }
	

    
    // The original insertion sort is copied for your reference.
    public static void insertionSortOrig(Element[] a) {
        int i, j, n = a.length;
        Element key;
        for (i = 1; i < n; i++) {
            key = a[i];
            for (j = i - 1; j >= 0; j--) {
                if (a[j].compareTo(key) <= 0) break;     
                a[j + 1] = a[j];
            }
            a[j + 1] = key;
        }
    }
    
    public static void main(String[] args) {
        int[] input = {3, 10, 20, 3, 4, 5}; // try different inputs.
        int n = input.length;
        Element[] a = new Element[n];
        // Element[] a_orig = new Element[n];
        for (int i = 0; i < input.length; i++) {
            a[i] = new Element(i, input[i]);
            // a_orig[i] = new Element(i, input[i]);
		}
        
        System.out.println("Original: " + Arrays.toString(a));
        insertionSort(a);
        System.out.println("After sorted: \n" + Arrays.toString(a));
        //insertionSortOrig(a_orig);
        //System.out.println("After sorted (Ori): \n" + Arrays.toString(a_orig));
    }
}

