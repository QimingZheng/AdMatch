#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

#include <iostream>
#include <vector>
#include "src/common.h"

using namespace std;

// Return true of the input char is a word character [A-Za-z0-9_]
__device__ bool is_word_char(unsigned char c);

__device__ int blockReduceSum(int val);

// show matching results, including final active states, accepted rules
// called when SHOW_RESULTS is on
void show_results(int array_size, vector<ST_T> *final_states,
                  vector<int> *accept_rules);

// show profiling results
// called when PROFILER_MODE is on
void Profiler(struct timeval start_time, struct timeval end_time,
              int array_size, cudaEvent_t memalloc_start,
              cudaEvent_t memalloc_end, cudaEvent_t memcpy_h2d_start,
              cudaEvent_t memcpy_h2d_end, cudaEvent_t kernel_start,
              cudaEvent_t kernel_end, cudaEvent_t memcpy_d2h_start,
              cudaEvent_t memcpy_d2h_end, cudaEvent_t memfree_start,
              cudaEvent_t memfree_end);

#endif
