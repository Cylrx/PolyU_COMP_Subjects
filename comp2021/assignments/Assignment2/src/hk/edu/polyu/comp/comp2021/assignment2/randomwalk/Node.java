package hk.edu.polyu.comp.comp2021.assignment2.randomwalk;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

class Node{

    // degree of a node is the number of adjacency nodes, i.e., the number of nodes that are connected to this node by an edge.
    private int degree;
    //The graph this node belongs to
    private Graph graph;
    private HashMap<Node, Integer> out;

    public Graph getGraph(){
        return this.graph;
    }

    public void setGraph(Graph graph){
        this.graph = graph;
    }

    public void setDegree(){
        // Task 3.1: Calculate this.degree based on the random walk sequences.
        out = new HashMap<>();
        HashSet<RandomWalkSequence> rs = graph.getAllRandomWalkSequences();
        HashSet<Node> vis = new HashSet<>();
        for (RandomWalkSequence r: rs) {
            ArrayList<Node> seq = r.getSequence();
            int n = seq.size();
            for (int i = 0; i < n; i++) {
                if (seq.get(i) == this) {
                    if (i != n - 1) {
                        incr(seq.get(i + 1));
                        vis.add(seq.get(i + 1));
                    }
                    if (i != 0) vis.add(seq.get(i - 1));
                }
            }
        }
        degree = vis.size();
    }

    private void incr(Node x) {
        out.putIfAbsent(x, 0);
        out.put(x, out.get(x) + 1);
    }

    public int getDegree(){
        return this.degree;
    }

    public double transitionProbability(Node o){
        if(o == null){
            throw new IllegalArgumentException();
        }

        // Task 3.2: Given another node o, obtain the transition probability from this node to the given node.
        // transition probability is calculated by f(this, o) / f(this, all).
        // f(this, o) is the frequency of o as the next node of this within all random walk sequences.
        // f(this, all) is the frequency of this having a next node within all random walk sequences.
        // When f(this, all) = 0, the transition probability is 0.

        setDegree();

        double tot = 0;
        for (int freq: out.values()) tot += (double)freq;
        double ot = out.getOrDefault(o, 0);
        if (tot == 0) return 0;
        return (ot / tot);
    }
}