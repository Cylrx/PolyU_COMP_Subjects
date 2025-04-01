//
// scheduler.c
// The Scheduling Module (Part II)
//

#include <assert.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/wait.h>

#include "scheduler.h"
#include "input.h"
#include "vector.h"
#include "state.h"
#include "opti.h"

void process_request(Vector* queue, Request* req) {
    req->order = queue->size + 1; // order start from 1
    vector_add(queue, *req);
}

bool process_batch(Vector* queue, Request* req, int* invalid_cnt) {
    // read the batch file
    // parse the batch file
    // add the requests to the queue

    const char* file = req->file;
    FILE* fp = fopen(file, "r");

    if (fp == NULL) {
        printf("Error: Cannot open the batch file %s\n", file);
        return false;
    }
    
    while (!feof(fp)) {
        Request rq = file_input(fp);
        switch (rq.type) {
            case BATCH: {
                bool is_end = process_batch(queue, &rq, invalid_cnt);
                if (!is_end) break;
                __attribute__((fallthrough));
            }
            case TERMINATE:
                fclose(fp);
                return true;
            case REQUEST:
                process_request(queue, &rq);
                break;
            case PRINT:
                if (fork() == 0) {
                    schedule_and_print_bookings(req->algo, queue, *invalid_cnt);
                    exit(0);
                } else {
                    wait(NULL);
                }
                break;
            case INVALID:
                (*invalid_cnt)++;
                break;
            default:
                break;
        }
    }

    return false;
}

void run_fcfs(Vector* queue, Statistics* stats, Tracker* tracker) {
    for (int i = 0; i < queue->size; i++) {
        Request req = queue->data[i];
        int end = req.start + req.duration - 1;
        if (try_put(req.order, req.start, end, req.parking, req.essential, tracker)) { 
            vector_add(&stats->accepted, req);
        } else {
            vector_add(&stats->rejected, req);
        }
    }
}

void run_prio(Vector* queue, Statistics* stats, Tracker* tracker) {
    vector_qsort(queue, 0, queue->size-1, cmp_priority);
    for (int i = 0; i < queue->size; i++) {
        Request req = queue->data[i];
        int end = req.start + req.duration - 1;
        if (try_put(req.order, req.start, end, req.parking, req.essential, tracker)) { 
            vector_add(&stats->accepted, req);
        } else {
            vector_add(&stats->rejected, req);
        }
    }
}

// Simulated Annealing (SA) + Improved LJF Greedy Algorihthm.
void run_opti(Vector* queue, Statistics* stats, Tracker* tracker) {
    Vector* rejected = malloc(sizeof(Vector));
    Vector* accepted = malloc(sizeof(Vector));
    vector_init(accepted);
    vector_init(rejected);
    vector_overwrite(queue, rejected);

    opti_reset();
    double cur_util = 0.0;

    /* Start SA */

    while (opti_running()) {
        vector_qsort(rejected, 0, rejected->size - 1, cmp_volume_cnt);
        opti_delete(rejected, accepted, tracker);
        // printf("DEBUG: delete\n");
        opti_greedy(rejected, accepted, tracker, true);
        // printf("DEBUG: greedy\n");
        double tmp_util = opti_util(accepted);
        // printf("DEBUG: NEW UTIL = %.3lf\n", tmp_util);
        if (opti_accept(tmp_util, cur_util)) {
            cur_util = tmp_util;
            opti_backup(rejected, accepted, tracker);
            // printf("ACCEPTED\n");
            // debug_util(accepted);
        }
        else {
            opti_rollback(rejected, accepted, tracker);
            // printf("REJECTED, rolling back\n");
            // debug_util(accepted);
        }
        opti_iter();
    }
    
    vector_qsort(rejected, 0, rejected->size - 1, cmp_volume_cnt);
    opti_greedy(rejected, accepted, tracker, false);

    /* Free Memory */
    vector_overwrite(accepted, &stats->accepted);
    vector_overwrite(rejected, &stats->rejected);
    vector_free(accepted);
    vector_free(rejected);
    free(accepted);
    free(rejected);

}