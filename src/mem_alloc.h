#ifndef MEM_ALLOC_H
#define MEM_ALLOC_H

// Allocate host memory. Return true if the allocation succeeds.
bool alloc_host(void **ptr, size_t size);

// Free host memory
void free_host(void *ptr);

#endif