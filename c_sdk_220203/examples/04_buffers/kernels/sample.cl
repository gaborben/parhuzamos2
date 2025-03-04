__kernel void sample_kernel(__global int* buffer, int n)
{
    size_t i = get_global_id(0);
    if (i < n) {
        buffer[i] += i;
    }
}
