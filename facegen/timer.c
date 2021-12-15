#include "timer.h"

#include <time.h>

#define N 8

static struct timespec start_time[N];

void timer_start(int i) {
    clock_gettime(CLOCK_REALTIME, &start_time[i]);
}

double timer_stop(int i) {
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return (tp.tv_sec - start_time[i].tv_sec) + (tp.tv_nsec - start_time[i].tv_nsec) * 1e-9;
}
