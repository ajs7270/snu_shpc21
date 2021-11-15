#include "mat_mul.h"

#include <mpi.h>
#include "util.h"

// for debug
#include <stdio.h>

static float* A_part;
static float* B_copy;
static float* C_part;
static int mpi_rank, mpi_size;

void mat_mul_init(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  alloc_mat(&A_part, M/2, K);
  alloc_mat(&B_copy, K, N);
  alloc_mat(&C_part, M/2, N);
}

void mat_mul_finalize() {
  free_mat(A_part);
  free_mat(B_copy);
  free_mat(C_part);
}

void mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  /*****************************************************
   * SUPER INEFFICIENT MPI-BASED MATRIX MULTIPLICATION *
   *****************************************************/

  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; ++i) {
      MPI_Send(A, M * K / 2, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      MPI_Send(B, K * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(A_part, M * K / 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
    MPI_Recv(B_copy, K * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
  }

  // Hmm... Let's just calculate WHOLE matrix in ALL processes.
  if (mpi_rank == 0) {
    #pragma omp parallel for
		for(int i = M/2; i < M; ++i){
			for(int j = 0; j < N; ++j){
				C[i * N + j] = 0;
			}
		}

    #pragma omp parallel for
    for (int i = M/2; i < M; i+=2) {
      for (int k = 0; k < K; ++k) {
				float a0 = A[(i + 0) * K + k];
				float a1 = A[(i + 1) * K + k];
      	for (int j = 0; j < N; ++j) {
					float b = B[k * N + j];
					float c0 = C[(i + 0) * N + j];
					float c1 = C[(i + 1) * N + j];
					C[(i + 0) * N + j] = a0 * b + c0;
					C[(i + 1) * N + j] = a1 * b + c1;
        }
      }
    }
  } else {
		#pragma omp parallel for
		for(int i = 0; i < M/2; ++i){
			for(int j = 0; j < N; ++j){
				C_part[i * N + j] = 0;
			}
		}

		#pragma omp parallel for
    for (int i = 0; i < M/2; i+=2) {
      for (int k = 0; k < K; ++k) {
				float a0 = A_part[(i + 0) * K + k];
				float a1 = A_part[(i + 1) * K + k];
      	for (int j = 0; j < N; ++j) {
					float b = B_copy[k * N + j];
					float c0 = C_part[(i + 0) * N + j];
					float c1 = C_part[(i + 1) * N + j];
					C_part[(i + 0) * N + j] = a0 * b + c0;
					C_part[(i + 1) * N + j] = a1 * b + c1;
        }
      }
    }
  }

  // Hmm... Let's just receive WHOLE matrices from ALL processes.
  if (mpi_rank == 0) {
    MPI_Recv(C, M * N / 2, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, NULL);
  } else {
    MPI_Send(C_part, M * N / 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
}
