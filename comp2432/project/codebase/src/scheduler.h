//
// scheduler.h
// The Scheduler
//

#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "output.h"
#include "segtree.h"
#include "utils.h"
#include "state.h"

#include <stdbool.h>

void process_request(Vector* queue, Request* req);
bool process_batch(Vector* queue, Request* req, int* invalid_cnt);

// void run_all(Vector* queue[], Statistics* stats[], Tracker* trackers []);
void run_fcfs(Vector* queue, Statistics* stats, Tracker* tracker);
void run_prio(Vector* queue, Statistics* stats, Tracker* tracker);
void run_opti(Vector* queue, Statistics* stats, Tracker* tracker);


#endif //SCHEDULER_H
