#ifndef SEGTREE_H
#define SEGTREE_H

/**
 * Opaque handle representing K parallel segment trees.
 * Use public interface below for interactions.
 */
typedef struct SegTree SegTree;

/**
 * Creates K parallel segment trees covering [start, end]
 * 
 * @param start     Start index of all segment trees (inclusive)
 * @param end       End index of all segment trees (inclusive)
 * @param K         Number of parallel segment tree instances
 * @return          New segment tree handle. Must be destroyed with 
 *                  segtree_destroy(). NULL on allocation failure
 */
SegTree* segtree_build(int start, int end, unsigned K);

/**
 * Safely destroys segment trees and releases all resources
 * 
 * @param st        Handle from segtree_build(). No-op if NULL
 */
void segtree_destroy(SegTree* st);

/**
 * Resets all segment trees to empty state
 * @param st        Pointer to the segment tree to be emptied
 */
void segtree_empty(SegTree* st);

/**
 * Updates a specific segment tree instance within the collection
 * 
 * @param st        Handle from segtree_build()
 * @param k         Index of target segment tree (0 ≤ k < K)
 * @param l         Left range boundary to update (inclusive)
 * @param r         Right range boundary to update (inclusive)
 * @param val       Value to set for all elements in [l, r]
 */
void segtree_range_set(SegTree* st, unsigned k, int l, int r, int val);

/**
 * Queries all K instances for the maximum value in [l, r]
 *                  Stores the maximum value in the range for each k-th tree.
 * 
 * @param st        Handle from segtree_build()
 * @param l         Left query boundary (inclusive)
 * @param r         Right query boundary (inclusive)
 * @param results   Pre-allocated output array (size ≥ K).
 */
void segtree_range_query(SegTree* st, int l, int r, int* results);

/**
 * Overwrites the contents of target segment tree with source
 * 
 * @param source    Source segment tree to copy from
 * @param target    Target segment tree to copy to
 */
void segtree_overwrite(SegTree* source, SegTree* target);

#endif // SEGTREE_H
