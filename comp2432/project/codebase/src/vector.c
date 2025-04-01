#include "vector.h"
#include <assert.h>
#include <stdlib.h>

#define INIT_CAPACITY 20

void vector_init(Vector *vec) {
    vec->data = malloc(INIT_CAPACITY * sizeof(Request));
    vec->capacity = INIT_CAPACITY;
    vec->size = 0;
}

void vector_add(Vector *vec, Request req) {
    if (vec->size >= vec->capacity) {
        int new_capacity = vec->capacity == 0 ? 1 : vec->capacity * 2;
        Request *new_data = realloc(vec->data, (size_t)new_capacity * sizeof(Request));
        if (!new_data) exit(1);
        vec->data = new_data;
        vec->capacity = new_capacity;
    }
    vec->data[vec->size++] = req;
}

void vector_free(Vector *vec) {
    free(vec->data);
    vec->data = NULL;
    vec->size = vec->capacity = 0;
}

void vector_deepfree(Vector *vec) {
    for (int i = 0; i < vec->size; i++) {
        free(vec->data[i].file);
        free(vec->data[i].algo);
    }
    vector_free(vec);
}

static int is_work_hour(int start, int end) {
    // check if [start, end] within 8:00 AM ~ 8:00 PM
    assert(start <= end);
    int start_h = (start % (24 * 60)) / 60;
    int end_h = (end % (24 * 60)) / 60;
    return (start_h >= 8 && end_h < 20) ? 1 : 0;
}

int cmp_priority(const void *a, const void *b) {
    const Request *ra = (const Request*)a;
    const Request *rb = (const Request*)b;
    int prio_diff = ra->priority - rb->priority;
    if (prio_diff != 0) return prio_diff;
    int is_work_a = is_work_hour(ra->start, ra->start + ra->duration - 1);
    int is_work_b = is_work_hour(rb->start, rb->start + rb->duration - 1);
    return is_work_b - is_work_a;
}

int cmp_duration(const void *a, const void *b) {
    const Request *ra = (const Request*)a;
    const Request *rb = (const Request*)b;
    return ra->duration - rb->duration;
}

int cmp_volume_cnt(const void *a, const void *b) {
    const Request *ra = (const Request*)a;
    const Request *rb = (const Request*)b;
    int a_cnt = 0, b_cnt = 0;
    a_cnt += (ra->parking) ? 1 : 0;
    b_cnt += (rb->parking) ? 1 : 0;
    for (int i = 0; i < 3; i++) {
        int mask = (1 << (2 - i));
        if (ra->essential & mask) a_cnt++;
        if (rb->essential & mask) b_cnt++;
    }
    int tmp = rb->duration * b_cnt - ra->duration * a_cnt; 
    if (tmp != 0) return tmp;
    return b_cnt - a_cnt;
}


void vector_qsort(Vector *vec, int l, int r, int (*cmp)(const void*, const void*)) {
    assert (r - l + 1 >= 0);
    qsort(vec->data + l, (size_t)(r - l + 1), sizeof(Request), cmp);
}

void vector_overwrite(Vector* source, Vector* target) {
    vector_free(target);
    target->data = malloc((unsigned)source->capacity * sizeof(Request));
    target->capacity = source->capacity;
    for (int i = 0; i < source->size; i++) {
        vector_add(target, source->data[i]);
    }
}
