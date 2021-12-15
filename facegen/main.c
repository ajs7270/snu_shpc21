#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "timer.h"
#include "qdbmp.h"
#include "facegen.h"

const int NETWORK_SIZE_IN_BYTES = 20549132;

// read network binary
float* read_network(char *fn) {
    FILE *fin = fopen(fn, "rb");
    if (!fin) {
        fprintf(stderr, "Failed to open '%s'.\n", fn);
        exit(EXIT_FAILURE);
    }
    printf("Reading '%s'...", fn); fflush(stdout);

    fseek(fin, 0, SEEK_END);
    long sz = ftell(fin);
    if (sz != NETWORK_SIZE_IN_BYTES) {
        fprintf(stderr, "Expected %dB, but actual size is %ldB.\n", NETWORK_SIZE_IN_BYTES, sz);
        exit(EXIT_FAILURE);
    }
    fseek(fin, 0, SEEK_SET);

    float *network = (float*)malloc(NETWORK_SIZE_IN_BYTES);
    fread(network, 1, NETWORK_SIZE_IN_BYTES, fin);
    fclose(fin);
    printf(" done!\n");
    return network;
}

// read (number of faces, 100) matrix from input file
float* read_inputs(char *fn, int *num_to_gen) {
    FILE *fin = fopen(fn, "r");
    if (!fin) {
        fprintf(stderr, "Failed to open '%s'.\n", fn);
        exit(EXIT_FAILURE);
    }
    printf("Reading '%s'...", fn); fflush(stdout);

    fscanf(fin, "%d", num_to_gen);
    float *inputs = (float*)malloc(*num_to_gen * 100 * sizeof(float));
    for (int n = 0; n < *num_to_gen; ++n) {
        for (int i = 0; i < 100; ++i) {
            fscanf(fin, "%f", &inputs[n * 100 + i]);
        }
    }
    fclose(fin);
    printf(" done!\n");
    return inputs;
}

void write_outputs(char *fn, int num_to_gen, float *outputs) {
    printf("Writing '%s'...", fn); fflush(stdout);
    FILE *fout = fopen(fn, "w");
    fprintf(fout, "%d\n", num_to_gen);
    for (int i = 0; i < num_to_gen; ++i) {
        for (int j = 0; j < 64 * 64 * 3; ++j) {
            fprintf(fout, "%.4f ", outputs[i * 64 * 64 * 3 + j]);
        }
        fprintf(fout, "\n");
    }
    fclose(fout);
    printf(" done!\n");
}

// write result in sqaure tiled form
void write_image(char *fn, int num_to_gen, float *outputs) {
    printf("Writing '%s'...", fn); fflush(stdout);
    int sn = 0;
    for (; sn * sn < num_to_gen; ++sn);
    BMP *bmp = BMP_Create(sn * 64, sn * 64, 24);
    for (int sh = 0; sh < sn; ++sh) {
        for (int sw = 0; sw < sn; ++sw) {
            for (int h = 0; h < 64; ++h) {
                for (int w = 0; w < 64; ++w) {
                    int i = sh * 64 + h;
                    int j = sw * 64 + w;
                    int idx = sh * sn + sw;
                    if (idx >= num_to_gen) {
                        BMP_SetPixelRGB(bmp, j, i, 0, 0, 0);
                    } else {
                        // output has range (-1, 1), so transform to [0, 255]
                        float *p = &outputs[((idx * 64 + h) * 64 + w) * 3];
                        int r = (*p + 1) / 2 * 255;
                        int g = (*(p + 1) + 1) / 2 * 255;
                        int b = (*(p + 2) + 1) / 2 * 255;
                        BMP_SetPixelRGB(bmp, j, i, r, g, b);
                    }
                }
            }
        }
    }
    BMP_WriteFile(bmp, fn);
    BMP_Free(bmp);
    printf(" done!\n");
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <network binary> <input data> <output data> <output image>\n", argv[0]);
        fprintf(stderr, " e.g., %s network.bin input1.txt output1.txt output1.bmp\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    int num_to_gen;
    float *network = read_network(argv[1]);
    float *inputs = read_inputs(argv[2], &num_to_gen);
    float *outputs = (float*)malloc(num_to_gen * 64 * 64 * 3 * sizeof(float));

    // initialize; does not count into elapsed time
    printf("Initializing..."); fflush(stdout);
    facegen_init();
    printf(" done!\n");

    // main calculation
    printf("Calculating..."); fflush(stdout);
    timer_start(0);
    facegen(num_to_gen, network, inputs, outputs);
    double elapsed = timer_stop(0);
    printf(" done!\n");
    printf("Elapsed time : %.6f sec\n", elapsed);

    write_outputs(argv[3], num_to_gen, outputs);
    write_image(argv[4], num_to_gen, outputs);

    // finalize; does not count into elapsed time
    printf("Finalizing..."); fflush(stdout);
    facegen_fin();
    free(network);
    free(inputs);
    free(outputs);
    printf(" done!\n");

    return 0;
}
