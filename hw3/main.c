#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>

#include "timer.h"
#include "util.h"
#include "mat_mul.h"
#include "mat_mul_ref.h"

static void print_help(const char* prog_name) {
  printf("Usage: %s [-pvh] [-t num_threads] [-n num_iterations] [-w num_warmup] M N K\n", prog_name);
  printf("Options:\n");
  printf("  -p : print matrix data. (default: off)\n");
  printf("  -v : validate matrix multiplication. (default: off)\n");
  printf("  -h : print this page.\n");
  printf("  -t : number of threads (default: 1)\n");
  printf("  -n : number of iterations (default: 1)\n");
  printf("  -w : number of warmup iteration. (default: 0)\n");
  printf("   M : number of rows of matrix A and C. multiple of 16. (default: 16)\n");
  printf("   N : number of columns of matrix B and C. multiple of 16. (default: 16)\n");
  printf("   K : number of columns of matrix A and rows of B. multiple of 16. (default: 16)\n");
}

static bool print_matrix = false;
static bool validation = false;
static int M = 16, N = 16, K = 16;
static int num_threads = 1;
static int num_iterations = 1;
static int num_warmup = 0;

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
      case 't':
        num_threads = atoi(optarg);
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
        exit(0);
    }
  }
  for (int i = optind, j = 0; i < argc; ++i, ++j) {
    switch (j) {
      case 0:
        M = atoi(argv[i]);
        if (M % 16 != 0) {
          printf("M should be multiple of 16.\n");
          exit(0);
        }
        break;
      case 1:
        N = atoi(argv[i]);
        if (N % 16 != 0) {
          printf("N should be multiple of 16.\n");
          exit(0);
        }
        break;
      case 2:
        K = atoi(argv[i]);
        if (K % 16 != 0) {
          printf("K should be multiple of 16.\n");
          exit(0);
        }
        break;
      default:
        break;
    }
  }
  printf("Options:\n");
  printf("  Problem size: M = %d, N = %d, K = %d\n", M, N, K);
  printf("  Number of threads: %d\n", num_threads);
  printf("  Number of iterations: %d\n", num_iterations);
  printf("  Number of warmup iterations: %d\n", num_warmup);
  printf("  Print matrix: %s\n", print_matrix ? "on" : "off");
  printf("  Validation: %s\n", validation ? "on" : "off");
  printf("\n");
}

int main(int argc, char **argv) {
  timer_init(1);
  parse_opt(argc, argv);

  printf("Initializing... "); fflush(stdout);
  float *A, *B, *C;
  alloc_mat(&A, M, K);
  alloc_mat(&B, K, N);
  alloc_mat(&C, M, N);
  rand_mat(A, M, K);
  rand_mat(B, K, N);
  printf("done!\n");

  double elapsed_time_sum = 0;
  for (int i = -num_warmup; i < num_iterations; ++i) {
    if (i < 0) {
      printf("Warming up... "); fflush(stdout);
    } else {
      printf("Calculating...(iter=%d) ", i); fflush(stdout);
    }
    timer_reset(0);
    timer_start(0);
    mat_mul(A, B, C, M, N, K, num_threads);
    timer_stop(0);
    double elapsed_time = timer_read(0);
    printf("%f sec\n", elapsed_time);
    if (i >= 0) {
      elapsed_time_sum += elapsed_time;
    }
  }

  if (print_matrix) {
    printf("MATRIX A:\n"); print_mat(A, M, K);
    printf("MATRIX B:\n"); print_mat(B, K, N);
    printf("MATRIX C:\n"); print_mat(C, M, N);
  }

  if (validation) {
    float *C_ref;
    alloc_mat(&C_ref, M, N);
    timer_reset(0);
    timer_start(0);
    mat_mul_ref(A, B, C_ref, M, N, K, 16);
    timer_stop(0);
    double elapsed_time = timer_read(0);
    check_mat_mul(C, C_ref, M, N, K);
    printf("Reference time: %f sec\n", elapsed_time);
    printf("Reference throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time / 1e9);
  }

  double elapsed_time_avg = elapsed_time_sum / num_iterations;
  printf("Your Avg. time: %f sec\n", elapsed_time_avg);
  printf("Your Avg. throughput: %f GFLOPS\n", 2.0 * M * N * K / elapsed_time_avg / 1e9);

  timer_finalize();
  return 0;
}
