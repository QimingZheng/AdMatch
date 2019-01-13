#include "src/mem_alloc.h"

// Allocate host memory. Return true if the allocation succeeds.
bool alloc_host(void **ptr, size_t size) {
    return cudaMallocHost(ptr, size) == cudaSuccess;
}

// Free host memory
void free_host(void *ptr) { cudaFreeHost(ptr); }
