__kernel void sample_kernel(__global int* buffer, int n)
{
    size_t i = get_global_id(0);
    int k;
    int s;

    printf("global = %d, local = %d, group = %d\n", get_global_id(0), get_local_id(0), get_group_id(0));

    if (i > 0 && i < n - 1) {
        for (k = 0; k < (i % 5) + 1; ++k) {
            printf("k = %d\n", k);
            s = (buffer[i - 1] + buffer[i] + buffer[i + 1]) / 3;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        buffer[i] = s;
    }
}
