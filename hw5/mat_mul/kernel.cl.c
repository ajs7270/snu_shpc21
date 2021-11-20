__kernel void mat_mul(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  if (i >= M || j >= N) return;

  float s = 0;
  for (int k = 0; k < K; ++k) {
    s += A[i * K + k] * B[k * N + j];
  }
  C[i * N + j] = s;
}
