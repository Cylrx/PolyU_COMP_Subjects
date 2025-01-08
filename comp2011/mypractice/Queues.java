import java.util.Random;
import java.util.Scanner;

class SingleQueue {
	private int n, l, r;
	private int[] queue;

	public SingleQueue (int n) {
		this.n = n;
		this.l = -1;
		this.r = -1;
		this.queue = new int[n];
	}

	public boolean isEmpty () { return l == r; }
	public boolean isFull() { return r == n - 1; }
	public void enqueue (int x) { queue[++r] = x; }
	public int dequeue () { return queue[++l]; }
	
	public void printQueue () {
		for (int i = l + 1; i <= r; i++) System.out.print(queue[i] + " ");
		System.out.print(System.lineSeparator());
	}
}

class CircularQueue {
	private int n, l, r;
	private int[] q;

	public CircularQueue (int n) {
		this.n = n;
		this.l = 0;
		this.r = 0;
		this.q = new int[n];
	}

	public boolean isEmpty () { return l == r; }
	public boolean isFull () { return (r + 1) % n == l; }
	public void enqueue (int x) { q[r++] = x; r = r % n; }
	public int dequeue () {
		int res = q[l];
		q[l] = 0;
		l = (l + 1) % n;
		return res;
	}

	public void printQueue () {
		for (int x: q) System.out.print(x + " ");
		System.out.print(System.lineSeparator());
	}
}

public class Queues {
	public static void main (String[] args) {
		System.out.println("----Single Queue---");
		SingleQueue sq = new SingleQueue(20);
		sq.enqueue(3); sq.printQueue();
		sq.enqueue(4); sq.printQueue();
		System.out.println("dequeued " + sq.dequeue()); sq.printQueue();
		sq.enqueue(100); sq.printQueue();
		System.out.println("dequeued " + sq.dequeue()); sq.printQueue();

		System.out.println("---Circular Queue---");
		CircularQueue cq = new CircularQueue(5);
		while (true) {
			Scanner scan = new Scanner(System.in);
			String opcode = scan.next();
			if (opcode.equals("enq")) cq.enqueue(scan.nextInt());
			else if (opcode.equals("deq")) cq.dequeue();
			else if (opcode.equals("check")) {
				System.out.println("isFull = " + cq.isFull() + " | isEmpty = " + cq.isEmpty());
			} else System.out.println("Bad opcode");
			cq.printQueue();			
		}
	}
}
