#include "mat_mul.h"

#include <mpi.h>
#include "util.h"

static float* myA;
static float* myB;
static float* myC;
static int mpi_rank, mpi_size;

void mat_mul_init(int M, int N, int K) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  alloc_mat(&myA, M, K);
  alloc_mat(&myB, K, N);
  alloc_mat(&myC, M, N);
}

void mat_mul_finalize() {
  free_mat(myA);
  free_mat(myB);
  free_mat(myC);
}

void mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  /*****************************************************
   * SUPER INEFFICIENT MPI-BASED MATRIX MULTIPLICATION *
   *****************************************************/

  // Hmm... Let's just send WHOLE matrices to ALL processes.
  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; ++i) {
      MPI_Send(A, M * K, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
      MPI_Send(B, K * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(myA, M * K, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
    MPI_Recv(myB, K * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
  }

  // Hmm... Let's just calculate WHOLE matrix in ALL processes.
  #pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float s = 0;
      for (int k = 0; k < K; ++k) {
        s += myA[i * K + k] * myB[k * N + j];
      }
      myC[i * N + j] = s;
    }
  }

  // Hmm... Let's just receive WHOLE matrices from ALL processes.
  if (mpi_rank == 0) {
    for (int i = 1; i < mpi_size; ++i) {
      MPI_Recv(C, M * N, MPI_FLOAT, i, 0, MPI_COMM_WORLD, NULL);
    }
  } else {
    MPI_Send(myC, M * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }
}
