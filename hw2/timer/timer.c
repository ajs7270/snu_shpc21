#include "timer.h"
#define NULL 0
#define STOP 0
#define START 1

#include <stdlib.h>
#include <time.h>

struct timer {
    struct timespec start;
    struct timespec end;
    char flag;
};

struct timer * ts = NULL;

void timer_init(int n) {
    ts = (struct timer *)malloc(n*sizeof(struct timer));
    for(int i = 0; i < n; i++){
        ts[i].start.tv_sec = (long)0;
        ts[i].start.tv_nsec = (long)0;
        ts[i].end.tv_sec = (long)0;
        ts[i].end.tv_nsec = (long)0;
        ts[i].flag = STOP;
    }
}

void timer_finalize() {
    free(ts);
    ts = NULL;
}

void timer_start(int idx) {
    if (ts[idx].flag == START) {
        return;
    }
    struct timespec temp;
    clock_gettime(CLOCK_MONOTONIC, &temp);
    ts[idx].start.tv_sec = temp.tv_sec - (ts[idx].end.tv_sec - ts[idx].start.tv_sec);
    ts[idx].start.tv_nsec = temp.tv_nsec - (ts[idx].end.tv_nsec - ts[idx].start.tv_nsec);
    ts[idx].flag = START;
}

void timer_stop(int idx) {
    if (ts[idx].flag == STOP){
        return;
    }
    clock_gettime(CLOCK_MONOTONIC, &ts[idx].end);
    ts[idx].flag = STOP;
}

double timer_read(int idx) {
    if (ts[idx].flag == STOP){
        return ((ts[idx].end.tv_sec - ts[idx].start.tv_sec) + (ts[idx].end.tv_nsec - ts[idx].start.tv_nsec))/ (double)1000000000;
    }else{
        struct timespec temp;
        clock_gettime(CLOCK_MONOTONIC, &temp);
        return ((temp.tv_sec - ts[idx].start.tv_sec) + (temp.tv_nsec - ts[idx].start.tv_nsec)) / (double)1000000000;
    }
}

void timer_reset(int idx) {
    ts[idx].start.tv_sec = (long)0;
    ts[idx].start.tv_nsec = (long)0;
    ts[idx].end.tv_sec = (long)0;
    ts[idx].end.tv_nsec = (long)0;
    ts[idx].flag = STOP;
}
