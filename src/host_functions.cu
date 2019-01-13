#include <sys/time.h>
#include <algorithm>
#include <iostream>
#include "src/host_functions.h"
#include "src/mem_alloc.h"
#include "src/nfa_kernels.h"
#include "src/state_vector.h"

using namespace std;

// Host function to run iNFAnt algorithm on GPU
// This function can process multiple strings on a NFA simultaneously
// tg                   :  NFA transition graph
// h_input_array        :  array of input string in host memory
// input_bytes_array    :  array of string length
// array_size           :  array size (# of strings to match)
// threads_per_block    :  # of threads per block for kernel function
// show_match_result    :  print regex matching result if this variable is true
void run_nfa(struct ita_scratch &scratch, unsigned char **h_input_array,
             int *input_bytes_array, int array_size, int threads_per_block,
             bool show_match_result, bool profiler_mode,
             vector<int> *accepted_rules) {
    if (scratch.tg->kernel == iNFA)
        run_iNFA(scratch, h_input_array, input_bytes_array, array_size,
                 threads_per_block, show_match_result, profiler_mode,
                 accepted_rules);
    if (scratch.tg->kernel == TKO_NFA)
        run_TKO(scratch, h_input_array, input_bytes_array, array_size,
                threads_per_block, show_match_result, profiler_mode,
                accepted_rules);
    if (scratch.tg->kernel == AS_NFA)
        run_AS(scratch, h_input_array, input_bytes_array, array_size,
               threads_per_block, show_match_result, profiler_mode,
               accepted_rules);
}
