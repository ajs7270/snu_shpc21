#include "mat_mul.h"
#include "util.h"

#include <omp.h>

void mat_mul(float *A, float *B, float *C, int M, int N, int K, int num_threads) {
  // set num of threads
  omp_set_num_threads(num_threads);
  
  // initialize result matrix 
  #pragma omp parallel
  zero_mat(C, M, N);

  #pragma omp parallel
  {
		int r = 16;
		// parallelize matrix multiplication
		for (int i = 0; i < M; i++) {
#pragma omp parallel for schedule (auto)
			for (int j = 0; j < N; j += r) {
				for (int k = 0; k < K; k+= r) {
					//block matrix multiplication
					for (int jb = j; jb < j+r; jb++){
						for (int kb = k; kb < k+r; kb++){
							C[i * N + jb] += A[i*K + kb]*B[kb*N +jb];
						}
					}
				}
			}
		}
	}
}
