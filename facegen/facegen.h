#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void facegen_init();
void facegen(int num_to_gen, float *network, float *inputs, float *outputs);
void facegen_fin();

#ifdef __cplusplus
}
#endif
