#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define SAMPLE_SIZE 1 // Egy szám vizsgálata

typedef unsigned long long ull;

// Szekvenciális prímellenőrzés
int is_prime_sequential(ull n) {
    if (n < 2) return 0;
    if (n == 2 || n == 3) return 1;
    if (n % 2 == 0) return 0;
    for (ull i = 3; i <= sqrt(n); i += 2) {
        if (n % i == 0) return 0;
    }
    return 1;
}

int main(void) {
    cl_int err;
    int error_code;
    
    // Felhasználótól bekért szám
    ull num;
    printf("Enter a number to check for primality: ");
    scanf("%llu", &num);
    
    //ull num = 100000000000000003;
    
    
    printf("Checking number: %llu\n", num);

    // Szekvenciális futás időmérése
    clock_t start_seq = clock();
    int result_seq = is_prime_sequential(num);
    clock_t end_seq = clock();
    double time_seq = (double)(end_seq - start_seq) / CLOCKS_PER_SEC;

    printf("Sequential result for %llu: %s\n", num, result_seq ? "Prime" : "Not prime");
    printf("Sequential time: %f sec\n", time_seq);
    
    printf("Checking number (GPU): %llu\n", num);

    // Get platform
    cl_platform_id platform_id;
    cl_uint n_platforms;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        return 0;
    }

    // Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        return 0;
    }

    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
    const char* kernel_code = load_kernel_source("kernels/prime_check.cl", &error_code);
    if (error_code != 0) {
        printf("Source code loading error!\n");
        return 0;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        printf("Build error! Code: %d\n", err);
        printf("Build log: %s\n", build_log);
        free(build_log);
        return 0;
    }
    cl_kernel kernel = clCreateKernel(program, "is_prime_kernel", NULL);

    // Create OpenCL buffer
    cl_mem num_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(ull), NULL, NULL);
    cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&num_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&result_buffer);

    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(command_queue, num_buffer, CL_TRUE, 0, sizeof(ull), &num, 0, NULL, NULL);

    // Kernel execution and timing
    size_t global_work_size = 256;
    size_t local_work_size = 64;
    cl_event event;
    clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event);
    clFinish(command_queue);

    // Profiling information
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    double time_gpu = (end - start) / 1e9;

    // Retrieve result
    int result_gpu;
    clEnqueueReadBuffer(command_queue, result_buffer, CL_TRUE, 0, sizeof(int), &result_gpu, 0, NULL, NULL);

    printf("GPU result for %llu: %s\n", num, result_gpu ? "Prime" : "Not prime");
    printf("GPU time: %f sec\n", time_gpu);

    // Release resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);
    clReleaseMemObject(num_buffer);
    clReleaseMemObject(result_buffer);
    clReleaseCommandQueue(command_queue);

    return 0;
}
