__kernel void vec_add_normal_io(__global float *A,
                                __global float *B,
                                __global float *C
                                ) {
  int i = get_global_id(0);

  C[i] = A[i] + B[i];
}

__kernel void vec_add_vector_io(__global float *A,
                                __global float *B,
                                __global float *C,
                                int N) {
  int i = get_global_id(0);
  if (i >= N/16) return;

  float16 a = vload16(i, A);
  float16 b = vload16(i, B);
  float16 c = a + b;
  vstore16(c, i, C);
}
