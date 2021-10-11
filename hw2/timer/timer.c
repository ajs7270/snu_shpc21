#include "timer.h"
#define NULL	0
#define STOP	0
#define START	1
#define BILLION	1000000000	

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
        ts[i].start.tv_sec = (time_t)0;
        ts[i].start.tv_nsec = (time_t)0;
        ts[i].end.tv_sec = (time_t)0;
        ts[i].end.tv_nsec = (time_t)0;
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
    time_t t_sec, t_nsec;
    if (ts[idx].flag == STOP){
	t_sec = ts[idx].end.tv_sec - ts[idx].start.tv_sec;
	t_nsec = ts[idx].end.tv_nsec - ts[idx].start.tv_nsec;
    }else{
        struct timespec temp;
        clock_gettime(CLOCK_MONOTONIC, &temp);
	t_sec = temp.tv_sec - ts[idx].start.tv_sec;
	t_nsec = temp.tv_nsec - ts[idx].start.tv_nsec;
    }

    if(t_nsec < 0){
	t_nsec += BILLION;
    }

    return (t_sec + t_nsec)/(double)BILLION;
}

void timer_reset(int idx) {
    ts[idx].start.tv_sec = (time_t)0;
    ts[idx].start.tv_nsec = (time_t)0;
    ts[idx].end.tv_sec = (time_t)0;
    ts[idx].end.tv_nsec = (time_t)0;
    ts[idx].flag = STOP;
}
