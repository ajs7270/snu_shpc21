#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <mpi.h>

#include "timer.h"
#include "util.h"
#include "mat_mul.h"
#include "mat_mul_ref.h"

static bool print_matrix = false;
static bool validation = false;
static int M = 16, N = 16, K = 16;
static int num_iterations = 1;
static int num_warmup = 0;
static int mpi_rank, mpi_size;

static void print_help(const char* prog_name) {
  if (mpi_rank == 0) {
    printf("Usage: %s [-pvh] [-n num_iterations] [-w num_warmup] M N K\n", prog_name);
    printf("Options:\n");
    printf("  -p : print matrix data. (default: off)\n");
    printf("  -v : validate matrix multiplication. (default: off)\n");
    printf("  -h : print this page.\n");
    printf("  -n : number of iterations (default: 1)\n");
    printf("  -w : number of warmup iteration. (default: 0)\n");
    printf("   M : number of rows of matrix A and C. multiple of 16. (default: 16)\n");
    printf("   N : number of columns of matrix B and C. multiple of 16. (default: 16)\n");
    printf("   K : number of columns of matrix A and rows of B. multiple of 16. (default: 16)\n");
  }
}

static void parse_opt(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "pvht:n:w:")) != -1) {
    switch (c) {
      case 'p':
        print_matrix = true;
        break;
      case 'v':
        validation = true;
        break;
      case 'n':
        num_iterations = atoi(optarg);
        break;
      case 'w':
        num_warmup = atoi(optarg);
        break;
      case 'h':
      default:
        print_help(argv[0]);
        MPI_Finalize();
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0:
        M = atoi(argv[i]);
        if (M % 16 != 0) {
          printf("M should be multiple of 16.\n");
          MPI_Finalize();
          exit(0);
        }
        break;
      case 1:
        N = atoi(argv[i]);
        if (N % 16 != 0) {
          printf("N should be multiple of 16.\n");
          MPI_Finalize();
          exit(0);
        }
        break;
      case 2:
        K = atoi(argv[i]);
        if (K % 16 != 0) {
          printf("K should be multiple of 16.\n");
          MPI_Finalize();
          exit(0);
        }
        break;
      default:
        break;
    }
  }
  if (mpi_rank == 0) {
    printf("Options:\n");
    printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
    printf("  Number of iterations: %d\n", num_iterations);
    printf("  Number of warmup iterations: %d\n", num_warmup);
    printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
    printf("  Validation: %s\n", validation ? "on" : "off");
    printf("\n");
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  timer_init(1);
  parse_opt(argc, argv);

  float *A, *B, *C;
  if (mpi_rank == 0) {
    printf("[rank %d] Initializing matrices...\n", mpi_rank);
    alloc_mat(&A, M, K);
    alloc_mat(&B, K, N);
    alloc_mat(&C, M, N);
    rand_mat(A, M, K);
    rand_mat(B, K, N);
    printf("[rank %d] Initializing matrices done!\n", mpi_rank);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  printf("[rank %d] Initializing...\n", mpi_rank);
  mat_mul_init(M, N, K);
  printf("[rank %d] Initializing done!\n", mpi_rank);

  MPI_Barrier(MPI_COMM_WORLD);

  double elapsed_time_sum = 0;
  for (int i = -num_warmup; i < num_iterations; ++i) {
    if (i < 0) {
      printf("[rank %d] Warming up...\n", mpi_rank);
    } else {
      printf("[rank %d] Calculating...(iter=%d)\n", mpi_rank, i);
    }
    timer_reset(0);
    MPI_Barrier(MPI_COMM_WORLD);
    timer_start(0);
    mat_mul(A, B, C, M, N, K);
    MPI_Barrier(MPI_COMM_WORLD);
    timer_stop(0);
    double elapsed_time = timer_read(0);
    printf("[rank %d] %f sec\n", mpi_rank, elapsed_time);
    if (i >= 0) {
      elapsed_time_sum += elapsed_time;
    }
  }

  if (mpi_rank == 0) {
    if (print_matrix) {
      printf("MATRIX A:\n"); print_mat(A, M, K);
      printf("MATRIX B:\n"); print_mat(B, K, N);
      printf("MATRIX C:\n"); print_mat(C, M, N);
    }
  }

  if (mpi_rank == 0) {
    if (validation) {
      float *C_ref;
      alloc_mat(&C_ref, M, N);
      timer_reset(0);
      timer_start(0);
      mat_mul_ref(A, B, C_ref, M, N, K);
      timer_stop(0);
      double elapsed_time = timer_read(0);
      check_mat_mul(C, C_ref, M, N, K);
      free_mat(C_ref);
      printf("Reference time: %f sec\n", elapsed_time);
      printf("Reference throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time / 1e9);
    }
  }

  if (mpi_rank == 0) {
    free_mat(A);
    free_mat(B);
    free_mat(C);
  }

  if (mpi_rank == 0) {
    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Your Avg. time: %f sec\n", elapsed_time_avg);
    printf("Your Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);
  }

  printf("[rank %d] Finalizing...\n", mpi_rank);
  mat_mul_finalize();
  printf("[rank %d] Finalizing done!\n", mpi_rank);

  timer_finalize();
  MPI_Finalize();

  return 0;
}
