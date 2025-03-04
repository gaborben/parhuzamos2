__kernel void matrix_mul(__global float* A, __global float* B, __global float* C, int size) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
        printf("C[%d, %d] = %f\n", row, col, sum); // Debug
    }
}
