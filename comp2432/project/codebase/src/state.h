#ifndef STATE_H
#define STATE_H

#include "vector.h"
#include "segtree.h"

// The Statistics struct keeps track of accepted and rejected requests.
// Each scheduling algorithm should have exactly one Statistics instance.
// Run try_put() before modify this struct.
// If try_put() returns true, then add the corresponding request into the accepted Vector, else,
// add the request into the rejected Vector.
typedef struct Statistics {
    Vector accepted;
    Vector rejected;
} Statistics;


// The Tracker struct keeps track of the occupied time ranges for each resource.
// Each scheduling algorithm should have exactly one Tracker instance.
// This struct shall be modified in try_put() and try_essential().
typedef struct Tracker {
    SegTree* bc;    // battery + cable (0)
    SegTree* lu;    // locker + umbrella (1)
    SegTree* vi;    // valet parking + inflation services (2)
    SegTree* park;  // parking slots
} Tracker;

void init_tracker(Tracker* tr);
void init_statistics(Statistics* stats);
void reset_statistics(Statistics* stats);
void reset_tracker(Tracker* tr);
void tracker_overwrite(Tracker* source, Tracker* target);

#endif //STATE_H
