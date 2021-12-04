#include <stdio.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdlib.h>
#include <mpi.h>

#include "timer.h"
#include "util.h"
#include "vec_add.h"

// Usage: <program name> <vector size>

static int N = 209715200;
static int mpi_rank, mpi_size;

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
		rand_vec(A, N);
		rand_vec(B, N);
		printf("Initializing vectors done!\n");

		printf("Initializing...\n");
		vec_add_init(N);
		printf("Initializing done!\n");

		timer_reset(0);
		timer_start(0);
		vec_add(A, B, C, N);

		double elapsed_time = timer_read(0);
		printf("Elapsed time using normal I/O: %f sec\n", elapsed_time);
		printf("Reference throughput: %f GFLOPS\n", 2.0 * N / elapsed_time / 1e9);
	}

  printf("Finalizing...\n");
  vec_add_finalize();
  printf("Finalizing done!\n");

  free_vec(A);
  free_vec(B);
  free_vec(C);

  timer_finalize();
  MPI_Finalize();

  return 0;
}
