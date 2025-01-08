package hk.edu.polyu.comp.comp2021.cvfs.type;

import java.io.Serializable;

/**
 * Custom Pair object. Used to bundle two arbitrary objects together.
 *
 * @param <F> the type of the first object
 * @param <S> the type of the second object
 */
public class Pair<F, S> implements Serializable {
    private final F first;
    private final S second;

    /**
     * Creates a new Pair object with the given first and second objects.
     *
     * @param first the first object
     * @param second the second object
     */
    public Pair(F first, S second) {
        this.first = first;
        this.second = second;
    }

    /**
     * @return the first object
     */
    public F getFirst() { return first; }

    /**
     * @return the second object
     */
    public S getSecond() { return second; }
}
