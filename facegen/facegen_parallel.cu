#include <mpi.h>

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "facegen.h"
// for debugging
#define CHECK_CUDA(err) \
    do { \
        cudaError_t CHECK_CUDA_err = (err); \
        if (CHECK_CUDA_err != cudaSuccess) { \
            printf("[%s:%d] CUDA error %d (%s)\n", __FILE__, __LINE__, CHECK_CUDA_err, cudaGetErrorString(CHECK_CUDA_err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/*
 * Define global variables here.
 */

// for computation communication overlapping
#define NUM_OF_BUFFER 2 // for double buffering
#define NUM_OF_STREAM 3 
#define NUM_OF_FEATURE_MAP 4

extern const int NETWORK_SIZE_IN_BYTES;
extern int num_to_gen;

int num_to_gen_per_node;

// for MPI
static int mpi_rank, mpi_size;
static MPI_Request request;
static MPI_Request* nrequest;
static MPI_Status status;
static MPI_Status* nstatus;

// for CUDA Stream
static int inputStreamSize; 
static int inputStreamBytes;
static int outputStreamSize; 
static int outputStreamBytes;
static int featureMapSize[NUM_OF_FEATURE_MAP] = {4*4*512, 8*8*256, 16*16*128, 32*32*64};
static cudaStream_t stream[NUM_OF_STREAM];

// GPU Memory pointer
static float* gpu_inputs[NUM_OF_BUFFER]; // for double buffering 
static float* gpu_outputs[NUM_OF_BUFFER];
static float* gpu_network;
static float* gpu_fm[NUM_OF_FEATURE_MAP];

__global__ void proj(float *in, float *out, float *weight, float *bias, int C, int K) {

    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;

    float s = 0;
    for (int c = 0; c < C; ++c) {
        s += in[c] * weight[c * K + k];
    }
    s += bias[k];
    out[k] = s;
}

__global__ void batch_norm(float *inout, float *beta, float *gamma, float *mean, float *var, int HW, int C) {
    int hw = blockDim.x * blockIdx.x + threadIdx.x;
    if (hw >= HW) return;

    for (int c = 0; c < C; ++c) {
        float scaled_gamma = gamma[c] / sqrtf(var[c] + 1e-5);
        inout[hw * C + c] = scaled_gamma * inout[hw * C + c] + (beta[c] - scaled_gamma * mean[c]);
    }
}

__global__ void relu(float *inout, int HWC) {
    int hwc = blockDim.x * blockIdx.x + threadIdx.x;
    if (hwc >= HWC) return;

    inout[hwc] = fmaxf(inout[hwc], 0);
}

__global__ void tconv(float *in, float *out, float *weight, float *bias, int H_IN, int W_IN, int C, int K) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= K) return;

    int H_OUT = H_IN * 2, W_OUT = W_IN * 2;
    for (int h_out = 0; h_out < H_OUT; ++h_out) {
        for (int w_out = 0; w_out < W_OUT; ++w_out) {
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

__global__ void tanh_layer(float *inout, int HWC) {
    int hwc = blockDim.x * blockIdx.x + threadIdx.x;
    if (hwc >= HWC) return;
    inout[hwc] = tanhf(inout[hwc]);
}


void facegen_init() {
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int device_count;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(mpi_rank % device_count);

    // for non-blocking mpi send, receive
    nstatus = (MPI_Status *)malloc(mpi_size * sizeof(MPI_Status));
    nrequest = (MPI_Request *)malloc(mpi_size * sizeof(MPI_Request));

    num_to_gen_per_node = num_to_gen / mpi_size;

    // because num_to_gen_per_node can be fraction
    if (mpi_rank == 0){
        num_to_gen_per_node = num_to_gen - (num_to_gen_per_node * (mpi_size - 1));
    }

    // for double buffering
    inputStreamSize =  100;
    inputStreamBytes = inputStreamSize * sizeof(float);
    outputStreamSize = 64 * 64 * 3;
    outputStreamBytes = outputStreamSize * sizeof(float);

    // allocate input, output memory in the GPU
    for (int i = 0; i < NUM_OF_BUFFER; i++){
        CHECK_CUDA(cudaMalloc(&gpu_inputs[i], inputStreamBytes));
        CHECK_CUDA(cudaMalloc(&gpu_outputs[i], outputStreamBytes));
    }

    // allocate network memroy in the GPU
    CHECK_CUDA(cudaMalloc(&gpu_network, NETWORK_SIZE_IN_BYTES));

    // allocate feature map memroy in the GPU
    for(int i = 0; i < NUM_OF_FEATURE_MAP; i++){
        CHECK_CUDA(cudaMalloc(&gpu_fm[i], featureMapSize[i]*sizeof(float)));
    }

    // create stream
    for(int i = 0; i < NUM_OF_STREAM; i++){
        CHECK_CUDA(cudaStreamCreate(&stream[i]));
    }

    // synchronize
    CHECK_CUDA(cudaDeviceSynchronize());
    for(int i = 0; i < NUM_OF_STREAM; i++){
        CHECK_CUDA(cudaStreamSynchronize(stream[i]));
    }
}

void facegen(int num_to_gen, float *network, float *inputs, float *outputs) {
    /*
     * Host-to-devie memory copy,
     * CUDA kernel launch,
     * Device-to-host memory copy
     */

    // Send noise data to the each nodes
    if (mpi_rank == 0){
        int offset = inputStreamSize*(num_to_gen/mpi_size);
        float* mpi_inputs = inputs + inputStreamSize * num_to_gen_per_node; 
        for (int i = 1; i < mpi_size; i++){
            MPI_Send(mpi_inputs + (i-1)*offset, offset, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            MPI_Send(network, NETWORK_SIZE_IN_BYTES / sizeof(float), MPI_FLOAT, i, 0, MPI_COMM_WORLD);
        }
    }else{
        int offset = inputStreamSize*(num_to_gen/mpi_size);
        MPI_Recv(inputs, offset, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
        MPI_Recv(network, NETWORK_SIZE_IN_BYTES / sizeof(float), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, NULL);
    }

    // Send Network weight host to GPU
    CHECK_CUDA(cudaMemcpy(gpu_network, network, NETWORK_SIZE_IN_BYTES, cudaMemcpyHostToDevice));

    float *proj_w = gpu_network; gpu_network += 100 * 8192;
    float *proj_b = gpu_network; gpu_network += 8192;
    float *bn0_beta = gpu_network; gpu_network += 512;
    float *bn0_gamma = gpu_network; gpu_network += 512;
    float *bn0_mean = gpu_network; gpu_network += 512;
    float *bn0_var = gpu_network; gpu_network += 512;
    float *tconv1_w = gpu_network; gpu_network += 5 * 5 * 256 * 512;
    float *tconv1_b = gpu_network; gpu_network += 256;
    float *bn1_beta = gpu_network; gpu_network += 256;
    float *bn1_gamma = gpu_network; gpu_network += 256;
    float *bn1_mean = gpu_network; gpu_network += 256;
    float *bn1_var = gpu_network; gpu_network += 256;
    float *tconv2_w = gpu_network; gpu_network += 5 * 5 * 128 * 256;
    float *tconv2_b = gpu_network; gpu_network += 128;
    float *bn2_beta = gpu_network; gpu_network += 128;
    float *bn2_gamma = gpu_network; gpu_network += 128;
    float *bn2_mean = gpu_network; gpu_network += 128;
    float *bn2_var = gpu_network; gpu_network += 128;
    float *tconv3_w = gpu_network; gpu_network += 5 * 5 * 64 * 128;
    float *tconv3_b = gpu_network; gpu_network += 64;
    float *bn3_beta = gpu_network; gpu_network += 64;
    float *bn3_gamma = gpu_network; gpu_network += 64;
    float *bn3_mean = gpu_network; gpu_network += 64;
    float *bn3_var = gpu_network; gpu_network += 64;
    float *tconv4_w = gpu_network; gpu_network += 5 * 5 * 3 * 64;
    float *tconv4_b = gpu_network; gpu_network += 3;

    // run network for each face
    for (int n = 0; n < num_to_gen_per_node; ++n) {
        int input_offset = n * inputStreamSize;
        int output_offset = n * outputStreamSize;

        int buffer_index = n % NUM_OF_BUFFER;
        float *input = gpu_inputs[buffer_index];
        float *output = gpu_outputs[buffer_index];

        dim3 blockDim(32, 1, 1);
        dim3 gridDim(8192, 1, 1);
       
        // Noise input is currently on CPU. Send them to GPU.
        CHECK_CUDA(cudaMemcpyAsync(gpu_inputs[buffer_index], &inputs[input_offset], 
                    inputStreamBytes, cudaMemcpyHostToDevice, stream[buffer_index]));

        proj<<<gridDim, blockDim, 0, stream[buffer_index]>>>(input, gpu_fm[0], proj_w, proj_b, 100, 8192);
        // implicit layout change here; (8192,) -> (4, 4, 512)
        batch_norm<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[0], bn0_beta, bn0_gamma, bn0_mean, bn0_var, 4 * 4, 512);
        relu<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[0], 4 * 4 * 512);
        tconv<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[0], gpu_fm[1], tconv1_w, tconv1_b, 4, 4, 512, 256);
        batch_norm<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[1], bn1_beta, bn1_gamma, bn1_mean, bn1_var, 8 * 8, 256);
        relu<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[1], 8 * 8 * 256);
        tconv<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[1], gpu_fm[2], tconv2_w, tconv2_b, 8, 8, 256, 128);
        batch_norm<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[2], bn2_beta, bn2_gamma, bn2_mean, bn2_var, 16 * 16, 128);
        relu<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[2], 16 * 16 * 128);
        tconv<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[2], gpu_fm[3], tconv3_w, tconv3_b, 16, 16, 128, 64);
        batch_norm<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[3], bn3_beta, bn3_gamma, bn3_mean, bn3_var, 32 * 32, 64);
        relu<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[3], 32 * 32 * 64);
        tconv<<<gridDim, blockDim, 0, stream[buffer_index]>>>(gpu_fm[3],output, tconv4_w, tconv4_b, 32, 32, 64, 3);
        tanh_layer<<<gridDim, blockDim, 0, stream[buffer_index]>>>(output, 64 * 64 * 3);

        // Image output is currently on GPU. Send them to CPU.
        CHECK_CUDA(cudaMemcpyAsync(&outputs[output_offset], gpu_outputs[n%NUM_OF_BUFFER],
                    outputStreamBytes, cudaMemcpyDeviceToHost, stream[n%NUM_OF_BUFFER]));
    }

    for(int i = 0; i < NUM_OF_STREAM; i++){
        CHECK_CUDA(cudaStreamSynchronize(stream[i]));
    }

    // Recv ouput data from the each nodes
    if (mpi_rank == 0){
        int offset = outputStreamSize*(num_to_gen/mpi_size);
        float* mpi_outputs = outputs + outputStreamSize * num_to_gen_per_node; 
        for (int i = 1; i < mpi_size; i++){
            MPI_Irecv(mpi_outputs + (i-1)*offset, offset, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &nrequest[i]);
        }    
        for (int i = 1; i < mpi_size; i++){
            MPI_Wait(&nrequest[i], &nstatus[i]);
        }
    }else{
        int offset = outputStreamSize*(num_to_gen/mpi_size);
        MPI_Isend(outputs, offset, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request);
        MPI_Wait(&request, &status);
    }
}

void facegen_fin() {
    /*
     * Finalize required CUDA objects. For example,
     */
    for(int i = 0; i < NUM_OF_BUFFER; i++){
        CHECK_CUDA(cudaFree(gpu_inputs[i]));
        CHECK_CUDA(cudaFree(gpu_outputs[i]));
    }

    for(int i = 0; i < NUM_OF_FEATURE_MAP; i++){
        CHECK_CUDA(cudaFree(gpu_fm[i]));
    }

    for(int i = 0; i < NUM_OF_STREAM; i++){
        CHECK_CUDA(cudaStreamDestroy(stream[i]));
    }


}
