#include <mpi.h>

#include "vec_add.h"

#include "timer.h"
#include "util.h"
#include "cuda_util.h"


// for double buffering
#define NUM_OF_BUFFER 2
#define NUM_OF_STREAM 3

static float* gpu_mem_A[NUM_OF_BUFFER];
static float* gpu_mem_B[NUM_OF_BUFFER];
static float* gpu_mem_C[NUM_OF_BUFFER];
static int mpi_rank, mpi_size;
static int streamSize, streamBytes;
static MPI_Request requestA;
static MPI_Request requestB;
static MPI_Status status;
static cudaStream_t stream[NUM_OF_STREAM];

__global__ void add(float* A, float* B, float* C, int N){
	C[threadIdx.x] = A[threadIdx.x] + B[threadIdx.x];
}

void vec_add_init(int N) {
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
	streamSize =  N / NUM_OF_BUFFER;
	streamBytes = N / mpi_size / NUM_OF_BUFFER * sizeof(float);

	for(int i = 0; i < NUM_OF_BUFFER; i++){
		CHECK_CUDA(cudaMalloc(&gpu_mem_A[i], N * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&gpu_mem_B[i], N * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&gpu_mem_C[i], N * sizeof(float)));
	}

	for(int i = 0; i < NUM_OF_STREAM; i++){
		CHECK_CUDA(cudaStreamCreate(&stream[i]));
	}

	CHECK_CUDA(cudaDeviceSynchronize());
 }

void vec_add_finalize() {
  // Free all resources we allocated
	for(int i = 0; i < NUM_OF_BUFFER; i++){
		CHECK_CUDA(cudaFree(gpu_mem_A[i]));
		CHECK_CUDA(cudaFree(gpu_mem_B[i]));
		CHECK_CUDA(cudaFree(gpu_mem_C[i]));
	}

	for(int i = 0; i < NUM_OF_STREAM; i++){
		CHECK_CUDA(cudaStreamDestroy(stream[i]));
	}
}

void vec_add(float *A, float *B, float *C, int N) {
  // Matrix A and B is currently on CPU. Send them to GPU.
	if (mpi_rank == 0){
		for (int i = 1; i < mpi_size; i++){
			MPI_Isend(A + i*(N/mpi_size), N/mpi_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestA);
			MPI_Isend(B + i*(N/mpi_size), N/mpi_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestB);
		}
	}else{
			MPI_Irecv(A + mpi_rank*(N/mpi_size), N/mpi_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requestA);
			MPI_Irecv(B + mpi_rank*(N/mpi_size), N/mpi_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requestB);
			MPI_Wait(&requestA, &status);
			MPI_Wait(&requestB, &status);
	}


	for (int i = 0; i < NUM_OF_STREAM; i++){
		int offset = i * streamSize;
		CHECK_CUDA(cudaMemcpyAsync(&gpu_mem_A[i], &A[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]));
		CHECK_CUDA(cudaMemcpyAsync(&gpu_mem_B[i], &B[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]));
	}

	for (int i = 0; i < NUM_OF_STREAM; i++){
		int offset = i * streamSize;

		dim3 blockDim(32, 1, 1);
		dim3 gridDim(N, 1, 1);

		add<<<gridDim, blockDim, 0, stream[i]>>>(gpu_mem_A[offset], gpu_mem_B[offset], gpu_mem_C[offset], streamSize);
	}
	

	for (int i = 0; i < NUM_OF_STREAM; i++){
		int offset = i * streamSize;
		CHECK_CUDA(cudaMemcpyAsync(&A[offset], &gpu_mem_A[i], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
		CHECK_CUDA(cudaMemcpyAsync(&B[offset], &gpu_mem_B[i], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
	}

	CHECK_CUDA(cudaDeviceSynchronize());
}
