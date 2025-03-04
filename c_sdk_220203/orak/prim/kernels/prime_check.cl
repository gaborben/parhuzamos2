__kernel void is_prime_kernel(__global ulong* num, __global int* result) {
    ulong n = *num;
    int gid = get_global_id(0);
    int local_prime = 1;
    
    if (gid == 0) {
        *result = 1; // Alapértelmezésben prímnek tekintjük
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    if (n < 2) {
        local_prime = 0;
    } else if (n == 2) {
        local_prime = 1;
    } else if (n % 2 == 0) {
        local_prime = 0;
    } else {
        ulong sqrt_n = (ulong)sqrt((float)n); // OpenCL 1.2 kompatibilis sqrt használata
        for (ulong i = 3 + gid * 2; i <= sqrt_n; i += get_global_size(0) * 2) {
            if (n % i == 0) {
                local_prime = 0;
                break;
            }
        }
    }
    
    // Ha bármelyik szál talál osztót, akkor az eredményt nem prímre állítjuk
    if (local_prime == 0) {
        atomic_min(result, 0);
    }
}