#include "facegen.h"

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * linear projection (matrix-vector multiplication)
 * in : (C)
 * out : (K)
 * weight : (C, K)
 * bias : (K)
 */
static void proj(float *in, float *out, float *weight, float *bias, int C, int K) {
  for (int k = 0; k < K; ++k) {
    float s = 0;
    for (int c = 0; c < C; ++c) {
      s += in[c] * weight[c * K + k];
    }
    s += bias[k];
    out[k] = s;
  }
}

/*
 * batch normalization (in-place)
 * inout : (H, W, C)
 * beta : (C)
 * gamma : (C)
 * mean : (C)
 * var : (C)
 */
static void batch_norm(float *inout, float *beta, float *gamma, float *mean, float *var, int HW, int C) {
  for (int hw = 0; hw < HW; ++hw) {
    for (int c = 0; c < C; ++c) {
      float scaled_gamma = gamma[c] / sqrtf(var[c] + 1e-5);
      inout[hw * C + c] = scaled_gamma * inout[hw * C + c] + (beta[c] - scaled_gamma * mean[c]);
    }
  }
}

/*
 * ReLU (in-place)
 * inout : (H, W, C)
 */
static void relu(float *inout, int HWC) {
  for (int hwc = 0; hwc < HWC; ++hwc) {
    inout[hwc] = fmaxf(inout[hwc], 0);
  }
}

/*
 * transposed convolution
 * in : (H_IN, W_IN, C)
 * out : (H_IN * 2, W_IN * 2, K)
 * weight : (5, 5, K, C)
 * bias : (K)
 */
static void tconv(float *in, float *out, float *weight, float *bias, int H_IN, int W_IN, int C, int K) {
  int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
  for (int h_out = 0; h_out < H_OUT; ++h_out) {
    for (int w_out = 0; w_out < W_OUT; ++w_out) {
      for (int k = 0; k < K; ++k) {
        float ss = 0;
        for (int r = 0; r < 5; ++r) {
          for (int s = 0; s < 5; ++s) {
            // top and left side has padding 3, bottom and right side has padding 2
            // so subtract 3
            int h_in = h_out - 3 + r;
            int w_in = w_out - 3 + s;
            // stride is 2, so check coordinates fall into input element or empty space
            if (h_in % 2 == 0 && w_in % 2 == 0) {
              h_in /= 2;
              w_in /= 2;
              // boundary check
              if (0 <= h_in && h_in < H_IN && 0 <= w_in && w_in < W_IN) {
                for (int c = 0; c < C; ++c) {
                  // filter is stored in reverse; so use [4 - r][4 - s] instead of [r][s]
                  // ss += in[h_in][w_in][c] * weight[4 - r][4 - s][k][c];
                  ss += in[(h_in * W_IN + w_in) * C + c] * weight[(((4 - r) * 5 + (4 - s)) * K + k) * C + c];
                }
              }
            }
          }
        }
        ss += bias[k];
        // out[h_out][w_out][k] = ss;
        out[(h_out * W_OUT + w_out) * K + k] = ss;
      }
    }
  }
}

/*
 * tanh (in-place)
 * inout : (H, W, C)
 */
static void tanh_layer(float *inout, int HWC) {
  for (int hwc = 0; hwc < HWC; ++hwc) {
    inout[hwc] = tanhf(inout[hwc]);
  }
}

void facegen_init() {
}

void facegen(int num_to_gen, float *network, float *inputs, float *outputs) {
  // split network into each layer's parameter
  float *proj_w = network; network += 100 * 8192;
  float *proj_b = network; network += 8192;
  float *bn0_beta = network; network += 512;
  float *bn0_gamma = network; network += 512;
  float *bn0_mean = network; network += 512;
  float *bn0_var = network; network += 512;
  float *tconv1_w = network; network += 5 * 5 * 256 * 512;
  float *tconv1_b = network; network += 256;
  float *bn1_beta = network; network += 256;
  float *bn1_gamma = network; network += 256;
  float *bn1_mean = network; network += 256;
  float *bn1_var = network; network += 256;
  float *tconv2_w = network; network += 5 * 5 * 128 * 256;
  float *tconv2_b = network; network += 128;
  float *bn2_beta = network; network += 128;
  float *bn2_gamma = network; network += 128;
  float *bn2_mean = network; network += 128;
  float *bn2_var = network; network += 128;
  float *tconv3_w = network; network += 5 * 5 * 64 * 128;
  float *tconv3_b = network; network += 64;
  float *bn3_beta = network; network += 64;
  float *bn3_gamma = network; network += 64;
  float *bn3_mean = network; network += 64;
  float *bn3_var = network; network += 64;
  float *tconv4_w = network; network += 5 * 5 * 3 * 64;
  float *tconv4_b = network; network += 3;

  // intermediate buffer for feature maps
  float *fm0 = (float*)malloc(4 * 4 * 512 * sizeof(float));
  float *fm1 = (float*)malloc(8 * 8 * 256 * sizeof(float));
  float *fm2 = (float*)malloc(16 * 16 * 128 * sizeof(float));
  float *fm3 = (float*)malloc(32 * 32 * 64 * sizeof(float));

  // run network for each face
  for (int n = 0; n < num_to_gen; ++n) {
    float *input = inputs + n * 100;
    float *output = outputs + n * 64 * 64 * 3;
    proj(input, fm0, proj_w, proj_b, 100, 8192);
    // implicit layout change here; (8192,) -> (4, 4, 512)
    batch_norm(fm0, bn0_beta, bn0_gamma, bn0_mean, bn0_var, 4 * 4, 512);
    relu(fm0, 4 * 4 * 512);
    tconv(fm0, fm1, tconv1_w, tconv1_b, 4, 4, 512, 256);
    batch_norm(fm1, bn1_beta, bn1_gamma, bn1_mean, bn1_var, 8 * 8, 256);
    relu(fm1, 8 * 8 * 256);
    tconv(fm1, fm2, tconv2_w, tconv2_b, 8, 8, 256, 128);
    batch_norm(fm2, bn2_beta, bn2_gamma, bn2_mean, bn2_var, 16 * 16, 128);
    relu(fm2, 16 * 16 * 128);
    tconv(fm2, fm3, tconv3_w, tconv3_b, 16, 16, 128, 64);
    batch_norm(fm3, bn3_beta, bn3_gamma, bn3_mean, bn3_var, 32 * 32, 64);
    relu(fm3, 32 * 32 * 64);
    tconv(fm3, output, tconv4_w, tconv4_b, 32, 32, 64, 3);
    tanh_layer(output, 64 * 64 * 3);
  }

  // free resources
  free(fm0);
  free(fm1);
  free(fm2);
  free(fm3);
}

void facegen_fin() {
}
