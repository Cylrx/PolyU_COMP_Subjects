package assignment2;

import java.security.SecureRandom;
import java.util.Arrays;

/**
 * @author yixin cao (September 28, 2020)
 * <p>
 * Simulating a linked list with an array.
 */
public class ListOnArray {
    private int[] data;
    int size;

    /*
     * The constructor is slightly different to Lab 5.
     * But the difference is irrelevant to the task.
     */
    public ListOnArray(int n) {
        size = (n + 1) << 1;
        data = new int[size];

        data[0] = 0; // may be omitted in Java.
        for (int i = 2; i < size - 2; i += 2)
            data[i] = i + 1;
        data[size - 1] = 1;
        data[size - 2] = 0;
    }

    public boolean isEmpty() {
        return data[0] == 0;
    }

    public boolean isFull() {
        return data[size - 1] == 0;
    }

    public void err() {
        System.out.println("Oops...");
    }

    /*
     * Insert a new element as the new head of the list.
     */
    public void insertFirst(int x) {
        if (isFull()) {
            err();
            return;
        }
        int i = data[size - 1];
        data[size - 1] = data[i + 1];
        data[i + 1] = data[0];
        data[0] = i;
        data[i] = x;
    }

    /**
     * The *in-place* quicksort algorithm.
     * <p>
     * VERY IMPORTANT.
     * <p>
     * I've discussed this question with the following students:
     * 1. None
     * <p>
     * I've sought help from the following Internet resources and books:
     * 1. None
     * <p>
     * Running time: O( n^2 )
     * Average time: O( n log n ) ONLY IF the sequence is random with pair-wise distinct elements
     */

    private int quicksort1() {
        int i = data[0];
        int pivot = data[i], pivotPtr = data[0];

        int lBeg, lEnd = -1, rBeg, rEnd;

        while (data[i + 1] != 0) {
            int next = data[i + 1];
            if (data[next] < pivot) {
                if (lEnd == -1) lEnd = next;
                data[i + 1] = data[next + 1];
                data[next + 1] = data[0];
                data[0] = next;
            } else {
                i = data[i + 1];
            }
        }

        // sort right
        rEnd = pivotPtr;
        if (data[pivotPtr + 1] != 0) {
            lBeg = data[0]; // original beginning;
            data[0] = data[pivotPtr + 1];
            rEnd = quicksort1();
            rBeg = data[0]; // new beginning of the right after sorting right partition
            data[0] = lBeg;
            data[pivotPtr + 1] = rBeg;
        }

        // sort left
        if (lEnd != -1) {
            data[lEnd + 1] = 0;
            lEnd = quicksort1();
            if (lEnd != 0) {
                data[lEnd + 1] = pivotPtr;
            }
        }
        return rEnd;
    }

    public void quicksort() {
        if (data[0] == 0) return;
        quicksort1();
    }

    /*
     * Output the elements stored in the list in order.
     */
    public String toString() {
        if (isEmpty()) return "The list is empty.";
        StringBuilder sb = new StringBuilder();
        int i = data[0];
        sb.append(data[i++]);
        while (data[i] != 0) {
            i = data[i];
            sb.append(" -> ").append(data[i++]);
        }
        return sb.append('\u2193').toString();
    }


    /*
     * Todo: add at least ten more test cases to test your code.
     * Todo: modify the list (removing and adding elements) and sort it again.
     *
     * The use of data structures from Java libraries is allowed here and only here.
     */
    public void randomModify() {
        int i = data[0];
        SecureRandom rand = new SecureRandom();
        boolean added = false, deleted = false;
        while (i != 0) {
            if (!added && rand.nextDouble() < 0.5 && !isFull()) {
                int j = data[size - 1];
                data[size - 1] = data[j + 1];
                data[j] = rand.nextInt(-100, 100);
                data[j + 1] = data[i + 1];
                data[i + 1] = j;
                System.out.println("Modify the list: add " + data[j]);
                added = true;
            }
            else if (!deleted && rand.nextDouble() < 0.5 && data[i + 1] != 0 && !isEmpty()) {
                System.out.println("Modify the list: delete " + data[data[i + 1]]);
                data[i + 1] = data[data[i + 1] + 1];
                deleted = true;
            }
            i = data[i + 1];
        }
    }

    public static void main(String[] args) {
        // TODO1 Satisfied:
        int[][] testData = { // try different inputs.
                {},
                {1, 1, 1, 1, 1, 1, 1},
                {10, 8, -4, 89, 2, 0, 4, -19, 200},
                {5, 4, 3, 2, 1, 1, 0, 0, -1},
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                {1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8},
                {3, 2, 6, 13, 8, 4, 10, 7, 14, 11, 12, 5, 9},
                {65, 85, 17, 88, 66, 71, 45, 38, 95, 48, 18, 68, 60, 96, 55},
                {10, 8, 14, 89, 32, 50, 77, 38},
                {9, 1, 3, 45, 13, 8, 901, 43},
                {0, 0, 0, 999},
                {911, 32, 551, 900},
                {100, 123, 344, 1, 0, 0, 0, 112, 1, 3},
                {-1, -2, -3, -4},
                {100},
                {Integer.MAX_VALUE, Integer.MIN_VALUE, 0},
                {2, 4, 6, 8, 10, 12},
                {-10, 0, 10, 20},
                {7, 3, 5, 3, 7, 9},
                {12, -7, 45, 3, 0, 99},
                {1, 2},
                {50, 40, 30, 20, 10}
        };

        for (int[] a : testData) {
            int n = a.length;
            SecureRandom random = new SecureRandom();
            ListOnArray list = new ListOnArray(n + random.nextInt(1 + (n << 2)));  // you don't need to understand this line.
            System.out.println("-----------------");
            System.out.println("The original array: " + Arrays.toString(a));
            for (int i = n - 1; i >= 0; i--) list.insertFirst(a[i]);
            System.out.println("The original list: " + list);
            // You can uncomment the following line to print out how the elements are stored
            // System.out.println("The internal storage: " + Arrays.toString(list.data));
            list.quicksort();
            System.out.println("The sorted list: " + list);

            // TODO2 Satisfied (randomly add & delete linked list elements)
            list.randomModify();
            System.out.println("The modified list: " + list);
            list.quicksort();
            System.out.println("The re-sorted list: " + list);
        }

    }
}
