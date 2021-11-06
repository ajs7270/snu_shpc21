#include "mat_mul.h"

void mat_mul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float s = 0;
      for (int k = 0; k < K; ++k) {
        s += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = s;
    }
  }
}
