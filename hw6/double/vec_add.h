#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	void vec_add_init(int N);

	void vec_add_finalize();

	void vec_add(float *A, float *B, float *C, int N);
#ifdef __cplusplus
}
#endif

