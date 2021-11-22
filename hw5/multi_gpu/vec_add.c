#include "vec_add.h"

#include "timer.h"
#include "util.h"
#include "opencl_util.h"

#define MAX_DEV 4

static cl_platform_id platform;
static cl_device_id device[MAX_DEV];
static cl_context context;
static cl_command_queue queue[MAX_DEV];
static cl_program program;
static cl_kernel kernel_normio[MAX_DEV];
//static cl_kernel kernel_vecio[MAX_DEV];
static cl_mem gpu_mem_A[MAX_DEV];
static cl_mem gpu_mem_B[MAX_DEV];
static cl_mem gpu_mem_C[MAX_DEV];
static unsigned int ndev = MAX_DEV;

void vec_add_init(int N) {
  cl_int err;

  // Get OpenCL platform
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_OPENCL(err);
  print_platform_info(platform);

  // Get num of device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL,(unsigned int*) &ndev);
  CHECK_OPENCL(err);

  // Get OpenCL device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ndev, device, NULL);
  CHECK_OPENCL(err);
  for(int i = 0; i < ndev; i++){
    print_device_info(device[i]);
  }

  // Create OpenCL context
  context = clCreateContext(NULL, ndev, device, NULL, NULL, &err);
  CHECK_OPENCL(err);

  // Create OpenCL command queue
  for(int i = 0; i < ndev; i++){
    queue[i] = clCreateCommandQueue(context, device[i], 0, &err);
    CHECK_OPENCL(err);
  }

  /*
   * Compile OpenCL program from "kernel.cl.c"
   * The name of kernel file is usually "kernel.cl",
   * but appending ".c" to the end of the filename helps text editors' syntax-highlighting.
   */
  program = create_and_build_program_with_source(context, ndev, device, "kernel.cl.c");

  for(int i = 0; i < ndev; i++){
    kernel_normio[i] = clCreateKernel(program, "vec_add_normal_io", &err); 
    CHECK_OPENCL(err);
   // kernel_vecio[i] = clCreateKernel(program, "vec_add_vector_io", &err); 
   // CHECK_OPENCL(err);
  }

  // Create GPU buffers for vectors
  for(int i = 0; i < ndev; i++){
    gpu_mem_A[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float)/ndev, NULL, &err);
    CHECK_OPENCL(err);
    gpu_mem_B[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float)/ndev, NULL, &err);
    CHECK_OPENCL(err);
    gpu_mem_C[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float)/ndev, NULL, &err);
    CHECK_OPENCL(err);
  }

  for(int i = 0; i < ndev; i++){
    err = clFinish(queue[i]);
    CHECK_OPENCL(err);
  }
}

void vec_add_finalize() {
  // Free all resources we allocated
  printf("ajs: pass %d\n", __LINE__);
  for(int i = 0; i < ndev; i++){
    clReleaseMemObject(gpu_mem_A[i]);
    clReleaseMemObject(gpu_mem_B[i]);
    clReleaseMemObject(gpu_mem_C[i]);
    clReleaseKernel(kernel_normio[i]);
    //clReleaseKernel(kernel_vecio[i]);
    clReleaseCommandQueue(queue[i]);
  }
  printf("ajs: pass %d\n", __LINE__);
  clReleaseProgram(program);
  clReleaseContext(context);
  printf("ajs: pass %d\n", __LINE__);
}

void vec_add(float *A, float *B, float *C, int N) {
  cl_int err;

  printf("ajs: pass %d\n", __LINE__);
  // Setup kernel arguments
  for(int i = 0; i < ndev; i++){
    err = clSetKernelArg(kernel_normio[i], 0, sizeof(cl_mem), &gpu_mem_A[i]);
    CHECK_OPENCL(err);
    err = clSetKernelArg(kernel_normio[i], 1, sizeof(cl_mem), &gpu_mem_B[i]);
    CHECK_OPENCL(err);
    err = clSetKernelArg(kernel_normio[i], 2, sizeof(cl_mem), &gpu_mem_C[i]);
    CHECK_OPENCL(err);

    /*
    err = clSetKernelArg(kernel_vecio[i], 0, sizeof(cl_mem), &gpu_mem_A[i]);
    CHECK_OPENCL(err);
    err = clSetKernelArg(kernel_vecio[i], 1, sizeof(cl_mem), &gpu_mem_B[i]);
    CHECK_OPENCL(err);
    err = clSetKernelArg(kernel_vecio[i], 2, sizeof(cl_mem), &gpu_mem_C[i]);
    CHECK_OPENCL(err);
    err = clSetKernelArg(kernel_vecio[i], 3, sizeof(int), &N);
    CHECK_OPENCL(err);
    */
  }

  // Vector A and B is currently on CPU. Send them to GPU.
  for(int i = 0; i < ndev; i++){
    err = clEnqueueWriteBuffer(queue[i], gpu_mem_A[i], CL_TRUE, 0, N * sizeof(float)/ndev,
		    (void*)((size_t)A + (N/ndev)*i), 0, NULL, NULL);
    CHECK_OPENCL(err);
    err = clEnqueueWriteBuffer(queue[i], gpu_mem_B[i], CL_TRUE, 0, N * sizeof(float)/ndev,
		    (void*)((size_t)B + (N/ndev)*i), 0, NULL, NULL);
    CHECK_OPENCL(err);
  }

  // Setup OpenCL global work size and local work size
  size_t gws[1] = {N/ndev}, lws[1] = {128};
  for (int i = 0; i < 1; ++i) {
    gws[i] = (gws[i] + lws[i] - 1) / lws[i] * lws[i];
  }

  // warm up
  for(int i = 0; i < ndev; i++){
    err = clEnqueueNDRangeKernel(queue[i], kernel_normio[i], 1, NULL, gws, lws, 0, NULL, NULL);
    CHECK_OPENCL(err);
  }

  for(int i = 0; i < ndev; i++){
    err = clFinish(queue[i]);
    CHECK_OPENCL(err);
  }

  // Run kernels
  timer_start(0);
  for(int i = 0; i < ndev; i++){
    err = clEnqueueNDRangeKernel(queue[i], kernel_normio[i], 1, NULL, gws, lws, 0, NULL, NULL);
    CHECK_OPENCL(err);
  }

  for(int i = 0; i < ndev; i++){
    err = clFinish(queue[i]);
    CHECK_OPENCL(err);
  }
  timer_stop(0);

  /*
  timer_start(1);
  err = clEnqueueNDRangeKernel(queue, kernel_vecio, 1, NULL, gws, lws, 0, NULL, NULL);
  CHECK_OPENCL(err);
  err = clFinish(queue);
  CHECK_OPENCL(err);
  timer_stop(1);
  */

  // After running kernel, result resides in gpu_mem_C. Send it back to CPU.
  for(int i = 0; i < ndev; i++){
    err = clEnqueueReadBuffer(queue[i], gpu_mem_C[i], CL_TRUE, 0, N * sizeof(float)/ndev, (void*)((size_t)C + (N/ndev)*i), 0, NULL, NULL);
    CHECK_OPENCL(err);
  }

  for(int i = 0; i < ndev; i++){
    err = clFinish(queue[i]);
    CHECK_OPENCL(err);
  }
}
