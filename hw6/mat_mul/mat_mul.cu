#include "mat_mul.h"

#include "timer.h"
#include "util.h"
#include "cuda_util.h"

static float* gpu_mem_A;
static float* gpu_mem_B;
static float* gpu_mem_C;

__global__ void sgemm(float* A, float* B, float* C, int M, int N, int K) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  if (i >= M || j >= N) return;

  float s = 0;
  for (int k = 0; k < K; ++k) {
    s += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = s;
}

void mat_mul_init(int M, int N, int K) {
  // Create GPU buffers for matrices
  CHECK_CUDA(cudaMalloc(&gpu_mem_A, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_mem_B, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gpu_mem_C, M * N * sizeof(float)));

  /*
   * DO NOT REMOVE BELOW LINES; NEEDED FOR CORRECT TIME MEASURE
   */
  CHECK_CUDA(cudaDeviceSynchronize());
}

void mat_mul_finalize() {
  // Free all resources we allocated
  CHECK_CUDA(cudaFree(gpu_mem_A));
  CHECK_CUDA(cudaFree(gpu_mem_B));
  CHECK_CUDA(cudaFree(gpu_mem_C));
}

void mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  // Matrix A and B is currently on CPU. Send them to GPU.
  CHECK_CUDA(cudaMemcpy(gpu_mem_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(gpu_mem_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  // Setup global work size and local work size
  dim3 blockDim(1, 32, 1);
  dim3 gridDim(M, N, 1);

  // Run kernel
  sgemm<<<gridDim, blockDim>>>(gpu_mem_A, gpu_mem_B, gpu_mem_C, M, N, K);

  // There is no way to check error from kernel launch syntax(i.e., <<< >>>). Check it here.
  CHECK_CUDA(cudaDeviceSynchronize());

  // After running kernel, result resides in gpu_mem_C. Send it back to CPU.
  CHECK_CUDA(cudaMemcpy(C, gpu_mem_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  /*
   * DO NOT REMOVE BELOW LINES; NEEDED FOR CORRECT TIME MEASURE
   */
  CHECK_CUDA(cudaDeviceSynchronize());
}
