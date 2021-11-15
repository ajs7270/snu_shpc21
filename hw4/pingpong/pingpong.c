#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <mpi.h>

#include "timer.h"

//#define NUM_OF_INT 1024*1024*128
#define NUM_OF_DOUBLE 10

static int mpi_rank, mpi_size;

int main(int argc, char **argv) {
	//int *data = (int*)malloc(NUM_OF_INT*sizeof(int)); // GB MB KB
	double data[NUM_OF_DOUBLE];
	int tag = 1001;

	MPI_Status status;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  printf("[rank %d] Ready to communicate...\n", mpi_rank);

  timer_init(1);
  timer_reset(0);

  MPI_Barrier(MPI_COMM_WORLD);
  timer_start(0);
  if (mpi_rank == 0) {
    MPI_Send(data, NUM_OF_DOUBLE, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD);
    // MPI_Recv...
  }
  else if (mpi_rank == 1) {
    MPI_Recv(data, NUM_OF_DOUBLE, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
    // MPI_Send...
    // MPI_Recv...
  }

  MPI_Barrier(MPI_COMM_WORLD);
  timer_stop(0);

  double elapsed_time = timer_read(0);
  if (mpi_rank == 0) {
    printf("Elapsed time: %f sec\n", elapsed_time);
  }

  timer_finalize();
  MPI_Finalize();

  return 0;
}
