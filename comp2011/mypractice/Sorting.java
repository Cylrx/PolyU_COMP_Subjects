import java.util.Random;
public class Sorting {
	static private Random rand = new Random();

	public static void main (String[] args) {
		System.out.println("--- merge sort ---");
		int[] arr = {1, 4, 8, 1, 3, 5, 0, 999};
		printArr(arr);
		mergeSort(arr, 0, arr.length - 1);
		printArr(arr);

		System.out.println("--- quick sort ---");
		int[] arr1 = {1, 4, 8, 1, 3, 5, 0, 999};
		printArr(arr1);
		quickSort(arr1, 0, arr1.length - 1);
		printArr(arr1);
	}

	private static void mergeSort (int[] arr, int l, int r) { if (l == r) return;
		int mid = (l + r) >> 1;
		mergeSort(arr, l, mid);
		mergeSort(arr, mid + 1, r);

		int[] b = new int[r - l + 1];
		int i = l, j = mid + 1, k = 0;
		while (i <= mid && j <= r) {
			while (i <= mid && arr[i] <= arr[j]) { b[k++] = arr[i]; i++; }
			if (i > mid) break;
			while (j <= r && arr[j] < arr[i]) { b[k++] = arr[j]; j++; }
			if (j > r) break;
		}

		for (; i <= mid; i++) b[k++] = arr[i];
		for (; j <= r; j++) b[k++] = arr[j];
		for (int z = l; z <= r; z++) arr[z] = b[z - l];
	}

	private static void quickSort (int[] arr, int l, int r) {
		if (l >= r) return;
		int z = rand.nextInt(r - l + 1) + l;
		int x = arr[z]; arr[z] = arr[r];

		int i = l, j = r;
		while (i < j) {
			while (i < j && arr[i] < x) i++;
			if (i < j) arr[j] = arr[i];
			while (i < j && arr[j] >= x) j--;
			if(i < j) arr[i] = arr[j];
		}
		arr[i] = x;
		
		quickSort(arr, l, i - 1);
		quickSort(arr, i + 1, r);
	}

	private static void radixSort(int[] arr) {
		int x = getMax(arr);
	}

	private static void getMax(int[] arr) {
		int max = Integer.MIN_VALUE;
		for (a
	}

	private static void printArr (int[] arr) {
		for (int i = 0; i < arr.length; i++) System.out.print(arr[i] + " ");
		System.out.println("\n");
	}
}
