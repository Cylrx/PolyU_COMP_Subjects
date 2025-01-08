public class LinkedList {
	public static void main (String[] args) {
		testDLL();
	}

	private static void testDLL () {
		System.out.println("----Double Linked List----");
		DLL dll = new DLL();
		System.out.println("\npushRear()");
		for (int i = 0; i < 3; i++) {
			dll.pushRear(i);
			dll.printList();
		}

		System.out.println("\npushHead()");
		for (int i = 4; i < 9; i++) {
			dll.pushHead(i);
			dll.printList();
		}

		System.out.println("\ndelete()");
		dll.delete(7); dll.printList();
		dll.delete(1); dll.printList();

		System.out.println("\ninsertBefore()");
		dll.insertBefore(2, 999);
		dll.printList();

		System.out.println("\ninsertAfter()");
		dll.insertAfter(2, 999);
		dll.printList();

		System.out.println("\nReverse()");
		dll.printList();
		dll.reverse();
		dll.printList();
	}
}

// Circular Double Linked List 
class CDLL {
	class Node {
		int v;
		Node prev, next;
	}
}


// Double Linked List
class DLL {
	class Node {
		int v;
		Node prev, next;
		Node (int v) { 
			this.v = v;
			prev = next = null;
		}
	}

	Node head, rear;
	DLL () { head = rear = null; }

	public void pushHead (int x) {
		if (isEmpty()) head = rear = new Node(x);
		else {
			Node tmp = new Node(x);
			tmp.next = head;
			head.prev = tmp;
			head = tmp;
		}
	}

	public void pushRear (int x) {
		if (isEmpty()) head = rear = new Node(x);
		else {
			Node tmp = new Node(x);
			tmp.prev = rear;
			rear.next = tmp;
			rear = tmp;
		}
	}

	public boolean insertBefore (int k, int x) {
		if (isEmpty()) return false;
		Node cur = head;
		while (cur.next != null && cur.next.v != k) cur = cur.next;
		if (cur == null) return false;

		Node tmp = new Node(x);
		tmp.prev = cur;
		tmp.next = cur.next;
		cur.next.prev = tmp;
		cur.next = tmp;
		return true;
	}
	
	public boolean insertAfter (int k, int x) {
		if (isEmpty()) return false;
		Node cur = head;
		while (cur != null && cur.v != k) cur = cur.next;
		if (cur == null) return false;

		Node tmp = new Node(x);
		tmp.prev = cur;
		tmp.next = cur.next;
		if (cur.next != null) cur.next.prev = tmp;
		else rear = tmp;
		cur.next = tmp;
		return true;
	}

	public boolean delete (int k) {
		Node cur = head;
		while (cur != null && cur.v != k) cur = cur.next;
		if (cur == null) return false;
		cur.prev.next = cur.next;
		cur.next.prev = cur.prev;
		return true;
	}

	public void printList () {
		Node cur = head;
		while (cur != null) {
			System.out.print(cur.v + " -> ");
			cur = cur.next;
		}
		System.out.println("null");
	}

	private boolean isEmpty() { return head == null && rear == null; }

	public void reverse () {
		if (isEmpty()) return;
		Node cur = head; 
		while (cur != null) {
			Node tmp = cur.next;
			cur.next = cur.prev;
			cur.prev = tmp;
			cur = tmp;
		}
		cur = head;
		head = rear;
		rear = cur;
	}
}
