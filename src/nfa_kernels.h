#ifndef NFA_KERNELS
#define NFA_KERNELS

#include <iostream>
#include <algorithm>
#include <sys/time.h>
#include "src/host_functions.h"

#include "src/transition_graph.h"
#include "src/common.h"
#include "src/kernel_common.h"

using namespace std;

#define block_ID blockIdx.x
#define thread_ID threadIdx.x
#define thread_count blockDim.x


void run_iNFA(class TransitionGraph *tg, unsigned char **h_input_array,
              int *input_bytes_array, int array_size, int threads_per_block,
              bool show_match_result, bool profiler_mode, vector<int> *accepted_rules);

void run_TKO(class TransitionGraph *tg, unsigned char **h_input_array,
             int *input_bytes_array, int array_size, int threads_per_block,
             bool show_match_result, bool profiler_mode, vector<int> *accepted_rules);

void run_AS(class TransitionGraph *tg, unsigned char **h_input_array,
            int *input_bytes_array, int array_size, int threads_per_block,
            bool show_match_result, bool profiler_mode, vector<int> *accepted_rules);

#endif
