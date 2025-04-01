#include "opti.h"
#include "segtree.h"
#include "vector.h"
#include "utils.h"
#include "rng.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

static int T_min, T_max;

/* SA Hyperparams */
static const double P = 0.9;
static const double Q = 0.3;
static const double INI_P = 0.99;
static const double END_P = 0.01;
static const int MAX_STEPS = 2000;

static Vector* pre_accepted = NULL;
static Vector* pre_rejected = NULL;
static Tracker* pre_tracker = NULL;

// the following needs reset everytime
double decay, cur_t;
int cur_step; 

static double fast_pow(double base, int exponent) {
    if (!exponent) return 1.0;
    int sign = exponent < 0;
    long exp = sign ? -(long)exponent : exponent;
    double res = 1.0;
    for (; exp; exp >>= 1, base *= base)
        if (exp & 1) res *= base;
    return sign ? 1.0 / res : res;
}

void opti_reset() {
    T_min = parse_time("2025-05-10", "00:00");
    T_max = parse_time("2025-05-16", "23:59");
    assert(T_min < T_max && T_max - T_min + 1 == 10080);
    assert(INI_P <= 1.0);

    if (pre_accepted) {
        vector_free(pre_accepted);
        vector_free(pre_rejected);
    } else {
        pre_accepted = malloc(sizeof(Vector));
        pre_rejected = malloc(sizeof(Vector));
        pre_tracker = malloc(sizeof(Tracker));
        vector_init(pre_accepted);
        vector_init(pre_rejected);
        init_tracker(pre_tracker);
    }

    double ref = -0.1;  // reference new_e - e
    double eps = 1e-9;
    double l = 1e-9, r = 1e9;

    // exp(ref / start_t) = INI_P
    // exp(ref / end_t) = END_P
    double start_t = ref / log(INI_P);
    double end_t = ref / log(END_P);

    while (r - l > eps) {
        decay = (r + l) / 2.0;
        if (start_t * fast_pow(decay, MAX_STEPS) > end_t) r = decay;
        else l = decay; // too fast
    }
    
    cur_t = start_t;
    cur_step = 0;
}

void opti_rollback(Vector* rejected, Vector* accepted, Tracker* tracker) {
    tracker_overwrite(pre_tracker, tracker);
    vector_overwrite(pre_accepted, accepted);
    vector_overwrite(pre_rejected, rejected);
}

void opti_backup(Vector* rejected, Vector* accpeted, Tracker* tracker) {
    tracker_overwrite(tracker, pre_tracker);
    vector_overwrite(rejected, pre_rejected);
    vector_overwrite(accpeted, pre_accepted);
}

void opti_greedy(Vector* rejected, Vector* accepted, Tracker* tracker, bool rand) {
    Vector* tmp_rejected = malloc(sizeof(Vector));
    vector_init(tmp_rejected);

    for (int i = 0; i < rejected->size; i++) {
        Request* req = &rejected->data[i];
        int end = req->start + req->duration - 1;
        if ((!rand || randd() < P) && try_put(req->order, req->start, end, req->parking, req->essential, tracker)) {
            vector_add(accepted, *req);
        } else {
            vector_add(tmp_rejected, *req);
        }
    }
    vector_overwrite(tmp_rejected, rejected);
}

void opti_delete(Vector* rejected, Vector* accepted, Tracker* tracker) {
    Vector* tmp_accepted = malloc(sizeof(Vector));
    vector_init(tmp_accepted);

    for (int i = 0; i < accepted->size; i++) {
        Request* req = &accepted->data[i];
        int end = req->start + req->duration - 1;
        if (randd() < Q) {
            try_delete(req->order, req->start, end, req->parking, req->essential, tracker);
            vector_add(rejected, *req);
        } else {
            vector_add(tmp_accepted, *req);
        }
    }
    vector_overwrite(tmp_accepted, accepted);
}

double opti_util(Vector* accepted) {
    int util = 0;
    for (int i = 0; i < accepted->size; i++) {
        Request* req = &accepted->data[i];
        int ess_cnt = ((req->essential & 0b100) ? 1 : 0) +
                      ((req->essential & 0b010) ? 1 : 0) +
                      ((req->essential & 0b001) ? 1 : 0);
        int park_cnt = req->parking ? 1 : 0;
        util += (req->duration) * (ess_cnt + park_cnt);
    }
    return (double)util / (double)((T_max - T_min + 1) * 19);
}

bool opti_accept(double new_util, double old_util) {
    if (new_util >= old_util) return true;
    double prob = exp((new_util - old_util) / cur_t);
    return (randd() < prob);
}

void opti_iter() {
    cur_step++;
    cur_t *= decay;
}

bool opti_running() {
    return cur_step < MAX_STEPS;
}




