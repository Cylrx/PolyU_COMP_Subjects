package assignment2;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * 
 * @author Yixin Cao (November 4, 2024)
 *
 * Use a maximum tree (introduced in Lecture 6) to sort an array.
 * 
 */
public class MaxTree { // Please change!

    /*
     * No modification to the class {@code Node} is allowed.
     * If you change anything in this class, your work will not be graded.
     */
    private class Node {
        int element;
        public Node leftChild, rightChild;
        public Node(int element) { this.element = element; }
        public String toString() { return String.valueOf(element); }
    }
    Node root;
    
    /**
     * The constructor: Build a maximum tree out of an array.
     *
     * VERY IMPORTANT.
     * 
     * I've discussed this question with the following students:
     *     1. none
     *
     * I've sought help from the following Internet resources and books:
     *     2. none
     *
     * Running time: O( n ).
     */ 
    public MaxTree(int[] a) {
        int n = a.length;
        Node[] c = new Node[n];
        for (int i = 0; i < n; i++) c[i] = new Node(a[i]);
        for (; n > 1; n = (n + 1) >> 1) {
            for (int i = 0; i < (n >> 1); i++) {
                Node tmp = new Node(Math.max(c[i * 2].element, c[i * 2 + 1].element));
                tmp.leftChild = c[i * 2];
                tmp.rightChild = c[i * 2 + 1];
                c[i] = tmp;
            }
            if ((n & 1) == 1) c[(n >> 1)] = c[n - 1];
        }
        this.root = c[0];
    }

    /**
     * Remove the root from a maximum tree and return its element.
     *
     * VERY IMPORTANT.
     * 
     * I've discussed this question with the following students:
     *     1. none
     *
     * I've sought help from the following Internet resources and books:
     *     1. none
     *
     * Running time: O( log n  ).
     */ 
    public int removeMax() {
        int ans = root.element;
        root = removeMax(root);
        return ans;
    }

    private Node removeMax(Node cur) {
        if (cur.leftChild == null || cur.rightChild == null) return null;
        if (cur.leftChild.element == cur.element) cur.leftChild = removeMax(cur.leftChild);
        else cur.rightChild = removeMax(cur.rightChild);

        if (cur.leftChild == null) cur = cur.rightChild;
        else if (cur.rightChild == null) cur = cur.leftChild;
        else cur.element = Math.max(cur.leftChild.element, cur.rightChild.element);
        return cur;
    }

	private void printTree() {
        if (root == null) return;
        Node[] q = new Node[1000];
        int l = 0, r = 0;
        q[r++] = root;

        ArrayList<Node> res = new ArrayList<>();
        while (l < r) {
            Node cur = q[l++];
            res.add(cur);
            if (cur.leftChild != null) q[r++] = cur.leftChild;
            if (cur.rightChild != null) q[r++] = cur.rightChild;
        }

        System.out.println(res);
        printer(res, 0, 1);
	}

    private void printer(ArrayList<Node> res, int s, int w) {
        int new_w = 0;
        if (s >= res.size()) return;
        StringBuilder branch = new StringBuilder();
        for (int i = s; i < s + w; i++) {
            Node cur = res.get(i);
            System.out.print(cur.element + " ");
            if (cur.leftChild != null) {
                new_w++;
                branch.append("|");
            } else branch.append(" ");
            branch.append(" ");
            if (cur.rightChild != null) {
                new_w++;
                branch.append("\\");
            } else branch.append(" ");
        }
        System.out.println("\n" + branch);
        printer(res, s + w, new_w);
    }

    /**
     * The *smart* selection sort.
     *
     * VERY IMPORTANT.
     * 
     * I've discussed this question with the following students:
     *     1. None
     *
     * I've sought help from the following Internet resources and books:
     *     1. None
     *
     * Running time: O( n log n  ); space use: O( n  ).
     */ 
    public static void treeSort(int[] a) {
        if (a.length == 0) return;
        MaxTree tree = new MaxTree(a);
        tree.printTree();
        for (int i = a.length - 1; i >= 0; i--) {
            a[i] = tree.removeMax();
            System.out.println("After removing " + a[i]);
            tree.printTree();
        }
    }

    /*
     * Todo: add at least ten more test cases to test your code.
     * The use of data structures from Java libraries is allowed here and only here.
     */
    public static void main(String[] args) {
        /*
        int[][] testData = {
                {},
                {1, 1, 1, 1, 1, 1, 1},
                {10, 8, -4, 89, 2, 0, 4, -19, 200},
                {5, 4, 3, 2, 1, 1, 0, 0, -1},
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                {1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8},
                {3, 2, 6, 13, 8, 4, 10, 7, 14, 11, 12, 5, 9},
                {65, 85, 17, 88, 66, 71, 45, 38, 95, 48, 18, 68, 60, 96, 55},
                {10, 8, 14, 89, 32, 50, 77, 38},
                {9, -9999},
                {100, 123, 344, 1, 0, 0, 0, 112, 1, 3},
                { -1, -2, -3, -4 },
                { 100 },
                { Integer.MAX_VALUE, Integer.MIN_VALUE, 0 },
                { 2, 4, 6, 8, 10, 12 },
                { -10, 0, 10, 20 },
                { 7, 3, 5, 3, 7, 9 },
                { 12, -7, 45, 3, 0, 99 },
                { 1, 2 },
                { 50, 40, 30, 20, 10 }
        };
         */
        int [][] testData = {
                {9, 0, 3, 1}
        };
        for (int[] a:testData) {
            System.out.println("The original array: " + Arrays.toString(a));
            treeSort(a);
            System.out.println("Sorted: " + Arrays.toString(a));
        }
    }
}
