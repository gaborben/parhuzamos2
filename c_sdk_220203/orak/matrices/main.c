#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 4  // Mátrix mérete (négyzetes mátrix esetén)

// Mátrix inicializálás véletlenszerű értékekkel
void initialize_matrix(float *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)(rand() % 10);
    }
}

// Mátrix kiírása
void print_matrix(float *matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%6.2f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    cl_int err;
    int error_code;

    // Mátrixok létrehozása és inicializálása
    float A[MATRIX_SIZE * MATRIX_SIZE];
    float B[MATRIX_SIZE * MATRIX_SIZE];
    float C[MATRIX_SIZE * MATRIX_SIZE] = {0};  // Inicializálás nullával

    initialize_matrix(A, MATRIX_SIZE);
    initialize_matrix(B, MATRIX_SIZE);

    printf("Matrix A:\n");
    print_matrix(A, MATRIX_SIZE);
    printf("Matrix B:\n");
    print_matrix(B, MATRIX_SIZE);

    // OpenCL inicializálás
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

    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Failed to create OpenCL context. Error code: %d\n", err);
        return 0;
    }

    // Kernel betöltése
    const char* kernel_code = load_kernel_source("kernels/matrix_mul.cl", &error_code);
    if (error_code != 0 || kernel_code == NULL) {
        printf("[ERROR] Source code loading error!\n");
        return 0;
    }

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Failed to create OpenCL program. Error code: %d\n", err);
        return 0;
    }

    err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
        printf("[ERROR] Kernel build failed! Code: %d\n", err);
        printf("Build log:\n%s\n", build_log);
        free(build_log);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Failed to create kernel. Error code: %d\n", err);
        return 0;
    }

    // OpenCL bufferek létrehozása
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), A, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), B, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), NULL, &err);

    if (!bufferA || !bufferB || !bufferC) {
        printf("[ERROR] Failed to create buffers!\n");
        return 0;
    }

    // Parancs sor létrehozása
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

    if (err != CL_SUCCESS) {
        printf("[ERROR] Failed to create command queue. Error code: %d\n", err);
        return 0;
    }

    // Kernel argumentumok beállítása
    int size = MATRIX_SIZE;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &size);

    // Kernel futtatása
    size_t global_work_size[2] = {MATRIX_SIZE, MATRIX_SIZE};
    cl_event event;
    err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Kernel execution failed! Error code: %d\n", err);
        return 0;
    }

    // Várakozás a végrehajtásra
    err = clFinish(command_queue);
    if (err != CL_SUCCESS) {
        printf("[ERROR] clFinish failed! Error code: %d\n", err);
        return 0;
    }

    // Eredmény visszaolvasása
    err = clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(float), C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Failed to read result buffer! Error code: %d\n", err);
        return 0;
    }

    printf("Result Matrix C:\n");
    print_matrix(C, MATRIX_SIZE);

    // Erőforrások felszabadítása
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseCommandQueue(command_queue);

    return 0;
}
