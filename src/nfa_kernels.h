#ifndef NFA_KERNELS_H
#define NFA_KERNELS_H

#include <sys/time.h>
#include <algorithm>
#include <iostream>
#include "src/host_functions.h"

#include "src/common.h"
#include "src/kernel_common.h"
#include "src/transition_graph.h"

using namespace std;

#define block_ID blockIdx.x
#define thread_ID threadIdx.x
#define thread_count blockDim.x

// run_iNFA will call iNFA kernel
void run_iNFA(struct ita_scratch &scratch, unsigned char **h_input_array,
              int *input_bytes_array, int array_size, int threads_per_block,
              bool show_match_result, bool profiler_mode,
              vector<int> *accepted_rules);

// run_TKO will call TKO kernel
void run_TKO(struct ita_scratch &scratch, unsigned char **h_input_array,
             int *input_bytes_array, int array_size, int threads_per_block,
             bool show_match_result, bool profiler_mode,
             vector<int> *accepted_rules);

// run_AS will call AS kernel
void run_AS(struct ita_scratch &scratch, unsigned char **h_input_array,
            int *input_bytes_array, int array_size, int threads_per_block,
            bool show_match_result, bool profiler_mode,
            vector<int> *accepted_rules);

#endif
