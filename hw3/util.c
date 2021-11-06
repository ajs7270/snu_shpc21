#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

void check_mat_mul(float *C, float *C_ref, int M, int N, int K) {
  printf("Validating...\n");

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float c = C[i * N + j];
      float c_ans = C_ref[i * N + j];
      if (fabsf(c - c_ans) > eps && (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
        ++cnt;
        if (cnt <= thr)
          printf("C[%d][%d] : correct_value = %f, your_value = %f\n", i, j, c_ans, c);
        if (cnt == thr + 1)
          printf("Too many error, only first %d values are printed.\n", thr);
        is_valid = false;
      }
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

void print_mat(float *m, int R, int C) {
  for (int i = 0; i < R; ++i) { 
    for (int j = 0; j < C; ++j) {
      printf("%+.3f ", m[i * C + j]);
    }
    printf("\n");
  }
}

void alloc_mat(float **m, int R, int C) {
  *m = (float *) aligned_alloc(64, sizeof(float) * R * C);
  if (*m == NULL) {
    printf("Failed to allocate memory for matrix.\n");
    exit(0);
  }
}

void rand_mat(float *m, int R, int C) {
  for (int i = 0; i < R; i++) { 
    for (int j = 0; j < C; j++) {
      m[i * C + j] = (float) rand() / RAND_MAX - 0.5;
    }
  }
}

void zero_mat(float *m, int R, int C) {
  memset(m, 0, sizeof(float) * R * C);
}
