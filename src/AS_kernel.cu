#include "src/nfa_kernels.h"

// iNFAnt traversal algorithm to process multiple strings on a NFA
// input                        :  total input string 
// input_offset                 :  offset of each input string
// transition_list              :  list of (source, destination) tuples
// transition_offset            :  index of first transition trigger by each symbol
// init_states_vector           :  vector of initial states 
// persis_states_vector         :  vector of persistent states
// final_states_vector          :  vector of final states
// vector_len                   :  length of state vector (# of ST_BLOCKs)   
__global__ void AS_kernel(unsigned char *input,
                           int *input_offset,
                           ST_BLOCK *transition_table,
                           Transition *transition_list,
                           ST_BLOCK *init_states_vector,
                           ST_BLOCK *final_states_vector,
                           int vector_len,
                           int state_count,
                           int wb_transition_count)
{       
        // Skip to the right input string 
        input += input_offset[block_ID];
        // Get the size of current input string
        int input_bytes = input_offset[block_ID + 1] - input_offset[block_ID];

        extern __shared__ ST_BLOCK s_data[];    // shared memory 

        ST_BLOCK *current_st_vec = s_data;                              // current active states in shared memory
        ST_BLOCK *future_st_vec = s_data + vector_len;                  // future active states in shared memory
        
        Transition tuple = transition_list[0];                
        ST_T src_state, dst_state;      
        ST_BLOCK src_bit, dst_bit;      
        unsigned int src_block, dst_block;
        int c;

        // Copy initial and persistent states from global memory into shared memory
        for (int i = thread_ID; i < vector_len; i += thread_count) {
                current_st_vec[i] = init_states_vector[i];
                future_st_vec[i] = 0;
        }

        __syncthreads();

        if (wb_transition_count == 0)
                goto BYPASS_HEAD;

        // If the first character is a word character, there is a word boundary before the first character
        if (!is_word_char(input[0]))
                goto BYPASS_HEAD;
        
        // For each transition triggered by word boundary 
        for (int i = thread_ID; i < wb_transition_count; i += thread_count) {
                tuple = transition_list[i];
                src_state = tuple.src;
                dst_state = tuple.dst;
                src_bit = 1 << (src_state % bit_sizeof(ST_BLOCK));      // index of state bit inside the block
                dst_bit = 1 << (dst_state % bit_sizeof(ST_BLOCK));
                src_block = src_state / bit_sizeof(ST_BLOCK);           // index of state block
                dst_block = dst_state / bit_sizeof(ST_BLOCK);
                
                // If transition source is set in current active state vector (divergence happens here)
                if (src_bit & current_st_vec[src_block]) {
                        // Set transition destination in CURRENT active state vector
                        atomicOr(&current_st_vec[dst_block], dst_bit);
                }
        }
        
        __syncthreads();

BYPASS_HEAD:
        // For each byte in the input string
        for (int byt = 0; byt < input_bytes; byt++) {

                for (int i = thread_ID; i < vector_len; i += thread_count) {
                        future_st_vec[i] = 0;
                }

                __syncthreads();

                c = (int)(input[byt]);
                // For each transition triggered by the character
                for(int blk = 0; blk <vector_len; blk ++){
                        int tmp = current_st_vec[blk];
                        if(tmp){
                                for (int s=blk*bit_sizeof(ST_BLOCK); s<min(
                                        (int)((blk+1)*bit_sizeof(ST_BLOCK)), (int)state_count); s++){
                                        if (tmp & (1<<(s % bit_sizeof(ST_BLOCK)))){
                                                for (int i = thread_ID; i < vector_len; i += thread_count) {
                                                        future_st_vec[i] |= transition_table[c*state_count*vector_len+s*vector_len+i];
                                                }
                                                __syncthreads();
                                        }
                                }
                        }
                }

                // Swap current and future active state vector
                if (current_st_vec == s_data) {
                        current_st_vec = s_data + vector_len;
                        future_st_vec = s_data;
                } else {
                        current_st_vec = s_data;
                        future_st_vec = s_data + vector_len;
                }

                __syncthreads();

                // No transition triggered by word boundary
                if (wb_transition_count == 0)
                        continue;

                // If there is NOT a word boundary between input[byt] and input[byt + 1] or after the last character
                if ((byt < input_bytes - 1 && (is_word_char(input[byt]) ^ is_word_char(input[byt + 1])) == 0) ||
                    (byt == input_bytes - 1 && !is_word_char(input[input_bytes - 1])))
                        continue;

                // For each transition triggered by word boundary
                for (int i = thread_ID; i < wb_transition_count; i += thread_count) {
                        tuple = transition_list[i];
                        src_state = tuple.src;
                        dst_state = tuple.dst;
                        src_bit = 1 << (src_state % bit_sizeof(ST_BLOCK));      // index of state bit inside the block
                        dst_bit = 1 << (dst_state % bit_sizeof(ST_BLOCK));
                        src_block = src_state / bit_sizeof(ST_BLOCK);           // index of state block
                        dst_block = dst_state / bit_sizeof(ST_BLOCK);
                
                        // If transition source is set in current active state vector (divergence happens here)
                        if (src_bit & current_st_vec[src_block]) {
                                // Set transition destination in CURRENT active state vector
                                atomicOr(&current_st_vec[dst_block], dst_bit);
                        }
                }
                __syncthreads();
        }

        // Copy final active states from shared memory into global memory
        for (int i = thread_ID; i < vector_len; i += thread_count) {
                final_states_vector[block_ID * vector_len + i] = current_st_vec[i];
        }
}

// Host function to run iNFAnt algorithm on GPU
// This function can process multiple strings on a NFA simultaneously
// tg                   :  NFA transition graph
// h_input_array        :  array of input string in host memory
// input_bytes_array    :  array of string length
// array_size           :  array size (# of strings to match)
// threads_per_block    :  # of threads per block for kernel function 
// show_match_result    :  print regex matching result if this variable is true                     
void run_AS(class TransitionGraph *tg, 
    unsigned char **h_input_array, 
    int *input_bytes_array, 
    int array_size,
    int threads_per_block, 
    bool show_match_result)
{
struct timeval start_time, end_time;
cudaEvent_t memalloc_start, memalloc_end;       // start and end events of device memory allocation
cudaEvent_t memcpy_h2d_start, memcpy_h2d_end;   // start and end events of memory copy from host to device
cudaEvent_t kernel_start, kernel_end;           // start and end events of kernel execution   
cudaEvent_t memcpy_d2h_start, memcpy_d2h_end;   // start and end events of memory copy from device to host
cudaEvent_t memfree_start, memfree_end;         // start and end events of device memory free

int vec_len = tg->init_states_vector.block_count;       // length (# of blocks) of state vector
int total_input_bytes = 0;                              // sum of string length

// Variables in host memory
unsigned char *h_input;                         // total input string  
int h_input_offset[array_size + 1];             // offsets of all input strings 
ST_BLOCK *h_final_st_vec;                       // final active states of all strings

// Variables in device memory
unsigned char *d_input;                                         // total input string
int *d_input_offset;                                            // offset of each input string
ST_BLOCK *d_init_st_vec, *d_final_st_vec;     // state vectors
ST_BLOCK *d_transition_table;
Transition *d_transition_list;

// Create events
cudaEventCreate(&memalloc_start);
cudaEventCreate(&memalloc_end);
cudaEventCreate(&memcpy_h2d_start);
cudaEventCreate(&memcpy_h2d_end);
cudaEventCreate(&kernel_start);
cudaEventCreate(&kernel_end);
cudaEventCreate(&memcpy_d2h_start);
cudaEventCreate(&memcpy_d2h_end);
cudaEventCreate(&memfree_start);
cudaEventCreate(&memfree_end);

gettimeofday(&start_time, NULL);

for (int i = 0; i < array_size; i++) {
       h_input_offset[i] = total_input_bytes;
       total_input_bytes += input_bytes_array[i];
}
h_input_offset[array_size] = total_input_bytes;

h_input = (unsigned char*)malloc(total_input_bytes);
if (!h_input) {
       cerr << "Error: allocate host memory to store total input string" << endl;
       return;
}

// Copy each string into h_input to construct a big string
for (int i = 0; i < array_size; i++) {
       memcpy(h_input + h_input_offset[i], h_input_array[i], input_bytes_array[i]);
}

// Allocate host memory
h_final_st_vec = (ST_BLOCK*)malloc(sizeof(ST_BLOCK) * vec_len * array_size);
if (!h_final_st_vec) {
       cerr << "Error: allocate host memory to store final state vectors" << endl;
       return;
}

// Allocate device memory
cudaEventRecord(memalloc_start, 0);
cudaMalloc((void **)&d_input, total_input_bytes);
cudaMalloc((void **)&d_input_offset, sizeof(int) * (array_size + 1));
cudaMalloc((void **)&d_init_st_vec, sizeof(ST_BLOCK) * vec_len);
cudaMalloc((void **)&d_final_st_vec, sizeof(ST_BLOCK) * vec_len * array_size);
cudaMalloc((void **)&d_transition_table, sizeof(ST_BLOCK) * vec_len * tg->state_count * SYMBOL_COUNT);
cudaMalloc((void **)&d_transition_list, sizeof(Transition) * tg->wb_transition_count);
cudaEventRecord(memalloc_end, 0);

// Copy input from host memory into device memory
cudaEventRecord(memcpy_h2d_start, 0);
cudaMemcpy(d_input, h_input, total_input_bytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_input_offset, h_input_offset, sizeof(int) * (array_size + 1), cudaMemcpyHostToDevice);
cudaMemcpy(d_init_st_vec, tg->init_states_vector.vector, sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
cudaMemcpy(d_transition_list, tg->transition_list, sizeof(Transition) * tg->wb_transition_count, cudaMemcpyHostToDevice);
for(int i=0;i<SYMBOL_COUNT;i++)
{
       for(int j=0;j<tg->state_count;j++)
       {
               cudaMemcpy(&d_transition_table[vec_len*(i*tg->state_count+j)],
                       tg->transition_table[i*tg->state_count+j].vector,
                       sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
       }
}
cudaEventRecord(memcpy_h2d_end, 0);

// Calculate the size of shared memory (for 3 state vectors and transition offset)
int shem = 2 * vec_len * sizeof(ST_BLOCK);

// Launch kernel
cudaEventRecord(kernel_start, 0);
AS_kernel<<<array_size, threads_per_block, shem>>>(d_input,
                                                   d_input_offset,
                                                   d_transition_table,
                                                   d_transition_list,
                                                   d_init_st_vec,
                                                   d_final_st_vec,
                                                   vec_len,
                                                   tg->state_count,
                                                   tg->wb_transition_count);
cudaEventRecord(kernel_end, 0);
cudaEventSynchronize(kernel_end);

// Copy result from device memory into host memory
cudaEventRecord(memcpy_d2h_start, 0);
cudaMemcpy(h_final_st_vec, d_final_st_vec, sizeof(ST_BLOCK) * vec_len * array_size, cudaMemcpyDeviceToHost);
cudaEventRecord(memcpy_d2h_end, 0);

// Get final active states and accept rules for each string
vector<ST_T> final_states[array_size];
vector<int> accept_rules[array_size];
unordered_map<ST_T, vector<int> >::iterator itr;

for (int i = 0; i < array_size; i++) {
       get_active_states(h_final_st_vec + i * vec_len, vec_len, final_states[i]);

       // Get all accept rules for string i
       for (int j = 0; j < final_states[i].size(); j++) {
               // Get accept rules triggered by this state
               itr = tg->accept_states_rules.find(final_states[i][j]);
               if (itr != tg->accept_states_rules.end()) {
                       accept_rules[i].insert(accept_rules[i].end(), itr->second.begin(), itr->second.end());
               } 
       }                

       // Remove repeated accept rules for string i
       sort(accept_rules[i].begin(), accept_rules[i].end());
       accept_rules[i].erase(unique(accept_rules[i].begin(), accept_rules[i].end() ), accept_rules[i].end()); 
       
} 

// Free device memory
cudaEventRecord(memfree_start, 0);
cudaFree(d_input);
cudaFree(d_input_offset);
cudaFree(d_init_st_vec);
cudaFree(d_final_st_vec);
cudaFree(d_transition_table);
cudaFree(d_transition_list);
cudaEventRecord(memfree_end, 0);

// Free host memory 
free(h_final_st_vec);
free(h_input);

gettimeofday(&end_time, NULL);

if (show_match_result) show_results(array_size, final_states, accept_rules);

Profiler(timeval start_time, 
        timeval end_time, 
        array_size, 
        memalloc_start, 
        memalloc_end,
        memcpy_h2d_start,
        memcpy_h2d_end,
        kernel_start,
        kernel_end,
        memcpy_d2h_start,
        memcpy_d2h_end,
        memfree_start,
        memfree_end
);

// Destroy events
cudaEventDestroy(memalloc_start);
cudaEventDestroy(memalloc_end);
cudaEventDestroy(memcpy_h2d_start);
cudaEventDestroy(memcpy_h2d_end);
cudaEventDestroy(kernel_start);
cudaEventDestroy(kernel_end);
cudaEventDestroy(memcpy_d2h_start);
cudaEventDestroy(memcpy_d2h_end);
cudaEventDestroy(memfree_start);
cudaEventDestroy(memfree_end);
}
