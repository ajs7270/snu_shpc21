#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <mpi.h>

#include "timer.h"
#include "util.h"
#include "vec_add.h"

// Usage: <program name> <vector size>

static long long N = 209715200;
static bool validation = false;
static bool print = false;
static int mpi_rank, mpi_size;
static int num_iterations = 1;
static int num_warmup = 1;


static void parse_opt(int argc, char **argv) {
  N = atoi(argv[1]);
  if (N % mpi_size * 2 != 0) {
    printf("N should be multiple of %d.\n", mpi_size);
    exit(0);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv); 

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  timer_init(9);
  parse_opt(argc, argv);


  float *A, *B, *C;
  if (mpi_rank == 0 ) {

    printf("Initializing vectors...\n");
    alloc_vec(&A, N);
    alloc_vec(&B, N);
    alloc_vec(&C, N);
    seq_vec(A, N);
    seq_vec_reverse(B, N);
    printf("Initializing vectors done!\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

    printf("Initializing...\n");
    vec_add_init(N);
    printf("Initializing done!\n");


    double elapsed_time_sum = 0;
    for (int i = -num_warmup; i < num_iterations; ++i) {
      if (i < 0) {
	printf("Warming up...\n");
      } else {
	printf("Calculating...(iter=%d)\n", i);
      }
      timer_reset(0);
      timer_start(0);
      vec_add(A, B, C, N);
      timer_stop(0);
      double elapsed_time = timer_read(0);
      if (i < 0) {
	printf("Warming up done!: %f sec\n", elapsed_time);
      } else {
	printf("Calculating done!(iter=%d): %f sec\n", i, elapsed_time);
      }
      if (i >= 0) {
	elapsed_time_sum += elapsed_time;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    if(print){
      print_vec('A',A, N);
      print_vec('B',B, N);
      print_vec('C',C, N);
    }

    double elapsed_time_avg = elapsed_time_sum / num_iterations;
    printf("Elapsed time using normal I/O: %f sec\n", elapsed_time_avg);
    printf("Reference throughput: %f GFLOPS\n", 2.0 * N / elapsed_time_avg / 1e9);

  printf("Finalizing...\n");
  vec_add_finalize();
  printf("Finalizing done!\n");

  if (mpi_rank == 0){
    check_vec(C, N);

    free_vec(A);
    free_vec(B);
    free_vec(C);
  }

  timer_finalize();
  MPI_Finalize();

  return 0;
}
