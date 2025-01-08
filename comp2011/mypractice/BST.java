public class BST {

	private static void insert(Node cur, int k) {
		if (cur.key > k) {
			if (cur.l == null) cur.l = new Node(k);
			else insert(cur.l, k);
		} else {
			if (cur.r == null) cur.r = new Node(k);
			else insert(cur.r, k);
		}
	}

	private static void inorder(Node cur) {
		if(cur == null) return;
		inorder(cur.l);
		System.out.print(cur.key + " ");
		inorder(cur.r);
	}

	private static Node delete (Node cur, int k) {
		if (cur.key < k) cur.r = delete(cur.r, k);
		else if (cur.key > k) cur.l = delete(cur.l, k);
		else {
			if (cur.r == null) return cur.r;
			if (cur.l == null) return cur.l;
			Node tmp = cur;
			cur = findMin(cur.r);
			cur.l = tmp.l;
			cur.r = tmp.r;
		}
		return cur;
	}

	private static Node findMin(Node cur) {
		if (cur.l != null) return findMin(cur.l);
		if (cur.r != null) return findMin(cur.r);
		return cur;
	}


	public static void main (String[] args){
		Node root = new Node(100);	
		insert(root, 10); insert(root, 101); insert(root, 9); insert(root, 11);
		inorder(root);

		delete(root, 101); inorder(root);
	}
}

class Node {
	int key;
	Node l = null, r = null;
	Node(int k) {
		this.key = k;
	}
}
