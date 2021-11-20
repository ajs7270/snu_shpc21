#include "mat_mul.h"

#include "timer.h"
#include "util.h"
#include "opencl_util.h"

static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_program program;
static cl_kernel kernel;
static cl_mem gpu_mem_A;
static cl_mem gpu_mem_B;
static cl_mem gpu_mem_C;

void mat_mul_init(int M, int N, int K) {
  cl_int err;

  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_OPENCL(err);
  print_platform_info(platform);

  // Get OpenCL device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_OPENCL(err);
  print_device_info(device);

  // Create OpenCL context
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_OPENCL(err);

  // Create OpenCL command queue
  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_OPENCL(err);

  /*
   * Compile OpenCL program from "kernel.cl.c"
   * The name of kernel file is usually "kernel.cl",
   * but appending ".c" to the end of the filename helps text editors' syntax-highlighting.
   */
  program = create_and_build_program_with_source(context, device, "kernel.cl.c");

  // Extract OpenCL kernel (i.e., a single function) from program
  kernel = clCreateKernel(program, "mat_mul", &err); 
  CHECK_OPENCL(err);

  // Create GPU buffers for matrices
  gpu_mem_A = clCreateBuffer(context, CL_MEM_READ_WRITE, M * K * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);
  gpu_mem_B = clCreateBuffer(context, CL_MEM_READ_WRITE, K * N * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);
  gpu_mem_C = clCreateBuffer(context, CL_MEM_READ_WRITE, M * N * sizeof(float), NULL, &err);
  CHECK_OPENCL(err);

  /*
   * DO NOT REMOVE BELOW 2 LINES; NEEDED FOR CORRECT TIME MEASURE
   */
  err = clFinish(queue);
  CHECK_OPENCL(err);
}

void mat_mul_finalize() {
  // Free all resources we allocated
  clReleaseMemObject(gpu_mem_A);
  clReleaseMemObject(gpu_mem_B);
  clReleaseMemObject(gpu_mem_C);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

void mat_mul(float *A, float *B, float *C, int M, int N, int K) {
  cl_int err;

  // Matrix A and B is currently on CPU. Send them to GPU.
  err = clEnqueueWriteBuffer(queue, gpu_mem_A, CL_FALSE, 0, M * K * sizeof(float), A, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clEnqueueWriteBuffer(queue, gpu_mem_B, CL_FALSE, 0, K * N * sizeof(float), B, 0, NULL, NULL);
  CHECK_OPENCL(err);

  // Setup kernel arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &gpu_mem_A);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &gpu_mem_B);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &gpu_mem_C);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &M); CHECK_OPENCL(err);
  err = clSetKernelArg(kernel, 4, sizeof(int), &N);
  CHECK_OPENCL(err);
  err = clSetKernelArg(kernel, 5, sizeof(int), &K);
  CHECK_OPENCL(err);

  // Setup OpenCL global work size and local work size
  size_t gws[2] = {M, N}, lws[2] = {8, 8};
  for (int i = 0; i < 2; ++i) {
    /*
     * By OpenCL spec, global work size should be MULTIPLE of local work size. The formula below achieves it.
     * e.g., gws = 25, lws = 16, then (25 + 16 - 1) / 16 * 16 = 40 / 16 * 16 = 2 * 16 = 32
     */
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // Run kernel
  err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, gws, lws, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clFinish(queue);
  CHECK_OPENCL(err);

  // After running kernel, result resides in gpu_mem_C. Send it back to CPU.
  err = clEnqueueReadBuffer(queue, gpu_mem_C, CL_TRUE, 0, M * N * sizeof(float), C, 0, NULL, NULL);
  CHECK_OPENCL(err);

  /*
   * DO NOT REMOVE BELOW 2 LINES; NEEDED FOR CORRECT TIME MEASURE
   */
  err = clFinish(queue);
  CHECK_OPENCL(err);
}
