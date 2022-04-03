#include <stdio.h>   // C programming header file
#include <unistd.h>  // C programming header file
#include <hello.cuh>

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code),
                file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif
__global__ void cuda_hello() { printf("Hello World from GPU!\n"); }
void wrap_hello(int a) {
    cuda_hello<<<1, a>>>();
    cudaDeviceSynchronize();
    return;
}
