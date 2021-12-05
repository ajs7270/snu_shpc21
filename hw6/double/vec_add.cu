#include <mpi.h>

#include "vec_add.h"
extern "C" {
#include "timer.h"
#include "util.h"
}
#include "cuda_util.h"


// for double buffering
#define NUM_OF_BUFFER 1
#define NUM_OF_STREAM 1

static float* A_part;
static float* B_part;
static float* C_part;

static float* gpu_mem_A[NUM_OF_BUFFER];
static float* gpu_mem_B[NUM_OF_BUFFER];
static float* gpu_mem_C[NUM_OF_BUFFER];
static int mpi_rank, mpi_size;
static int streamSize, streamBytes;
static MPI_Request requestA;
static MPI_Request requestB;
static MPI_Status status;
static MPI_Status* nstatus;
static MPI_Request* nrequest;
static cudaStream_t stream[NUM_OF_STREAM];

__global__ void add(float* A, float* B, float* C, int N){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  if (id < N) return;
  C[id] = A[id] + B[id];
}

void vec_add_init(int N) {
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  streamSize =  (N / mpi_size) / NUM_OF_BUFFER;
  streamBytes = streamSize * sizeof(float);

  // for non-blocking mpi send, receive
  nstatus = (MPI_Status *)malloc(mpi_size * sizeof(MPI_Status));
  nrequest = (MPI_Request *)malloc(mpi_size * sizeof(MPI_Request));

  if (mpi_rank != 0){
    alloc_vec(&A_part, N/mpi_size);
    alloc_vec(&B_part, N/mpi_size);
    alloc_vec(&C_part, N/mpi_size);
  }

  for(int i = 0; i < NUM_OF_BUFFER; i++){
    CHECK_CUDA(cudaMalloc(&gpu_mem_A[i], streamBytes));
    CHECK_CUDA(cudaMalloc(&gpu_mem_B[i], streamBytes));
    CHECK_CUDA(cudaMalloc(&gpu_mem_C[i], streamBytes));
  }

  for(int i = 0; i < NUM_OF_STREAM; i++){
    CHECK_CUDA(cudaStreamCreate(&stream[i]));
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  for(int i = 0; i < NUM_OF_STREAM; i++){
    CHECK_CUDA(cudaStreamSynchronize(stream[i]));
  }

  timer_init(9);
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
  if (mpi_rank == 0){
    for (int i = 1; i < mpi_size; i++){
      MPI_Isend(A + i*(N/mpi_size), N/mpi_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestA);
      MPI_Isend(B + i*(N/mpi_size), N/mpi_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requestB);
    }
  }else{
    MPI_Irecv(A_part, N/mpi_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requestA);
    MPI_Irecv(B_part, N/mpi_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requestB);
    MPI_Wait(&requestA, &status);
    A = A_part;
    B = B_part;
    C = C_part;
   MPI_Wait(&requestB, &status);
  }


  for (int i = 0; i < NUM_OF_BUFFER; i++){
    int offset = i * streamSize;
    int blockSize = 128;

    dim3 blockDim(blockSize, 1, 1);
    dim3 gridDim(streamSize/blockSize, 1, 1);

    // Matrix A and B is currently on CPU. Send them to GPU.
    CHECK_CUDA(cudaMemcpyAsync(gpu_mem_A[i], &A[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]));
    CHECK_CUDA(cudaMemcpyAsync(gpu_mem_B[i], &B[offset], streamBytes, cudaMemcpyHostToDevice, stream[i]));
    add<<<gridDim, blockDim, 0, stream[i]>>>(gpu_mem_A[i], gpu_mem_B[i], gpu_mem_C[i], offset);
    CHECK_CUDA(cudaMemcpyAsync(&C[offset], gpu_mem_C[i], streamBytes, cudaMemcpyDeviceToHost, stream[i]));
  }

  for(int i = 0; i < NUM_OF_STREAM; i++){
    CHECK_CUDA(cudaStreamSynchronize(stream[i]));
  }

  if (mpi_rank == 0){
    for (int i = 1; i < mpi_size; i++){
      MPI_Irecv(C + i*(N/mpi_size), N/mpi_size, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &nrequest[i]);
    }
    for (int i = 1; i < mpi_size; i++){
      MPI_Wait(&nrequest[i], &nstatus[i]);
    }
  }else{
    MPI_Isend(C, N/mpi_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &nrequest[mpi_rank]);
  }


}
