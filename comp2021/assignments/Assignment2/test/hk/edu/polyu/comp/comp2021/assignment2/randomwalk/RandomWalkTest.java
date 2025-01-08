package hk.edu.polyu.comp.comp2021.assignment2.randomwalk;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;

import static org.junit.jupiter.api.Assertions.*;

public class RandomWalkTest {

    public static float DELTA = 1E-6f;

    Graph g;
    Node n1, n2, n3, n4, n5, n6;
    RandomWalkSequence r1, r2, r3, r4, r5, r6;

    @BeforeEach
    public void prepare(){
        n1 = new Node();
        n2 = new Node();
        n3 = new Node();
        n4 = new Node();
        n5 = new Node();
        n6 = new Node();
        r1 = new RandomWalkSequence();
        r2 = new RandomWalkSequence();
        r3 = new RandomWalkSequence();
        r4 = new RandomWalkSequence();
        r5 = new RandomWalkSequence();
        r6 = new RandomWalkSequence();
        g = new Graph();
        n1.setGraph(g);
        n2.setGraph(g);
        n3.setGraph(g);
        n4.setGraph(g);
        n5.setGraph(g);
        n6.setGraph(g);

        ArrayList<Node> a1 = new ArrayList<Node>(Arrays.asList(n1, n6, n3, n2, n4, n5));
        ArrayList<Node> a2 = new ArrayList<Node>(Arrays.asList(n2, n1, n6, n5, n2, n4));
        ArrayList<Node> a3 = new ArrayList<Node>(Arrays.asList(n6, n3, n2, n1, n4, n6));
        ArrayList<Node> a4 = new ArrayList<Node>(Arrays.asList(n2, n1, n5, n2, n1, n4));
        ArrayList<Node> a5 = new ArrayList<Node>(Arrays.asList(n5, n3, n4, n2, n1, n3));
        ArrayList<Node> a6 = new ArrayList<Node>(Arrays.asList(n1, n2, n3, n4, n5, n6));

        r1.setSequence(a1);
        r2.setSequence(a2);
        r3.setSequence(a3);
        r4.setSequence(a4);
        r5.setSequence(a5);
        r6.setSequence(a6);

        HashSet<RandomWalkSequence> allRandomSequences = new HashSet<RandomWalkSequence>();
        allRandomSequences.add(r1);
        allRandomSequences.add(r2);
        allRandomSequences.add(r3);
        allRandomSequences.add(r4);
        allRandomSequences.add(r5);
        allRandomSequences.add(r6);

        g.setAllRandomWalkSequences(allRandomSequences);
    }


    @Test
    public void testSetDegree01() {
        n1.setDegree();
        assertEquals(5, n1.getDegree());
    }

    @Test
    public void testTransitionProbability01() {
        System.out.println(Math.abs(n3.transitionProbability(n2)));
        assert (Math.abs(n3.transitionProbability(n2) - 0.5) < DELTA);
    }

    @Test
    public void testOverlapping01() {
        HashSet<Node> h = new HashSet<Node>(Arrays.asList(n1, n2, n4, n3, n6, n5));
        assertTrue(h.equals(r1.nodeOverlapping(r6)));
    }

    @Test
    public void testEquals01() {
        assertFalse(r2.equals(r6));
    }

}
