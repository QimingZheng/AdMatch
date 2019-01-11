#ifndef KERNEL_COMMON
#define KERNEL_COMMON

#include <iostream>
#include "src/common.h"

using namespace std;

__device__ bool is_word_char(unsigned char c);

void show_results(int array_size, vector<ST_T> *final_states, vector<int> *accept_rules);

void Profiler(struct timeval start_time, 
              struct timeval end_time, 
              int array_size, 
              cudaEvent_t memalloc_start, 
              cudaEvent_t memalloc_end,
              cudaEvent_t memcpy_h2d_start,
              cudaEvent_t memcpy_h2d_end,
              cudaEvent_t kernel_start,
              cudaEvent_t kernel_end,
              cudaEvent_t memcpy_d2h_start,
              cudaEvent_t memcpy_d2h_end,
              cudaEvent_t memfree_start,
              cudaEvent_t memfree_end
)

#endif