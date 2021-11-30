#pragma once

void check_mat_mul(float *C, float *C_ref, int M, int N, int K);

void print_mat(float *m, int R, int C);

void alloc_mat(float **m, int R, int C);

void rand_mat(float *m, int R, int C);

void zero_mat(float *m, int R, int C);

void free_mat(float *m);
