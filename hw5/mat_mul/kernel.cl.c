__kernel void mat_mul(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  if (i >= M || j >= N) return;

  C[i * N + j] = 0;
  for (int k = 0; k < K; ++k) {
    C[i * N + j] += A[i * K + k] * B[k * N + j];
  }
}
