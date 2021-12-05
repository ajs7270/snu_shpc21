#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

void alloc_vec(float **m, int R) {
  *m = (float *) malloc(sizeof(float) * R);
  if (*m == NULL) {
    printf("Failed to allocate memory for vector.\n");
    exit(0);
  }
}

void rand_vec(float *m, int R) {
  for (int i = 0; i < R; i++) { 
    m[i] = (float) rand() / RAND_MAX - 0.5;
  }
}

void seq_vec(float *m, int R) {
  for (int i = 0; i < R; i++) { 
    m[i] = ((float)(i % 10001)/10000);
  }
}

void seq_vec_reverse(float *m, int R) {
  for (int i = 0; i < R; i++) { 
    m[i] = 1.0 - ((float)(i % 10001)/10000);
  }
}

void check_vec(float *m, int R){
	int flag = 1 ;
	for(int i = 0; i < R; i++){
		if (!(m[i] - 1.0 < 1/1e4 || m[i] - 1.0 > -1/1e4)){
			printf("ERROR : C[%d] = %f\n", i, m[i]);
			flag = 0;
		}
	}

	if(flag){
		printf("CORRECT VEC_ADD!!!\n");
	}
}

void print_vec(char name, float *m, int R){
	for(int i = 0; i < 3; i++){
		printf("PRINT : %c[%d] = %f\n",name, i, m[i]);
	}
}

void zero_vec(float *m, int R) {
  memset(m, 0, sizeof(float) * R);
}

void free_vec(float *m) {
  free(m);
}
