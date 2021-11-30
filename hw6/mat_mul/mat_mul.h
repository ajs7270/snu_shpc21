#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void mat_mul_init(int M, int N, int K);

void mat_mul_finalize();

void mat_mul(float *A, float *B, float *C, int M, int N, int K);

#ifdef __cplusplus
}
#endif
