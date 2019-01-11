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
__global__ void nfa_kernel(unsigned char *input,
                           int *input_offset,
                           Transition *transition_list,
                           int *transition_offset,
                           ST_BLOCK *init_states_vector,     
                           ST_BLOCK *persis_states_vector,
                           ST_BLOCK *final_states_vector,        
                           int vector_len)
{       
        // Skip to the right input string 
        input += input_offset[block_ID];
        // Get the size of current input string
        int input_bytes = input_offset[block_ID + 1] - input_offset[block_ID];

        extern __shared__ ST_BLOCK s_data[];    // shared memory 

        ST_BLOCK *current_st_vec = s_data;                              // current active states in shared memory
        ST_BLOCK *future_st_vec = s_data + vector_len;                  // future active states in shared memory
        ST_BLOCK *persis_st_vec = s_data + 2 * vector_len;              // persistent states in shared memory
        int *s_transition_offset = (int*)(s_data + 3 * vector_len);     // transition offset in shared memory

        Transition tuple = transition_list[0];                
        ST_T src_state, dst_state;      
        ST_BLOCK src_bit, dst_bit;      
        unsigned int src_block, dst_block;
        int c, transition_start, transition_count, wb_transition_start, wb_transition_count;

        // Copy initial and persistent states from global memory into shared memory
        for (int i = thread_ID; i < vector_len; i += thread_count) {
                current_st_vec[i] = init_states_vector[i];
                persis_st_vec[i] = persis_states_vector[i];
        }

        // Copy transition offset from global memory into shared memory
        // Note that there are SYMBOL_COUNT + 1 elements in total
        for (int i = thread_ID; i <= SYMBOL_COUNT; i += thread_count) {
                s_transition_offset[i] = transition_offset[i];
        }

        __syncthreads();

        // First transition and # of transitions triggered by word boundary
        wb_transition_start = s_transition_offset[WORD_BOUNDARY];
        wb_transition_count = s_transition_offset[WORD_BOUNDARY + 1] - wb_transition_start;
        
        if (wb_transition_count == 0)
                goto BYPASS_HEAD;

        // If the first character is a word character, there is a word boundary before the first character
        if (!is_word_char(input[0]))
                goto BYPASS_HEAD;
        
        // For each transition triggered by word boundary 
        for (int i = thread_ID; i < wb_transition_count; i += thread_count) {
                tuple = transition_list[i + wb_transition_start];
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
                // Keep persistent states active
                // future_st_vec <- current_st_vec & persis_st_vec
                for (int i = thread_ID; i < vector_len; i += thread_count) {
                        future_st_vec[i] = current_st_vec[i] & persis_st_vec[i];
                }
                __syncthreads();

                c = (int)(input[byt]);             
                transition_start = s_transition_offset[c];                        
                transition_count = s_transition_offset[c + 1] - transition_start; 

                // For each transition triggered by the character
                for (int i = thread_ID; i < transition_count; i += thread_count) {
                        tuple = transition_list[i + transition_start];
                        src_state = tuple.src;   
                        dst_state = tuple.dst;
                        src_bit = 1 << (src_state % bit_sizeof(ST_BLOCK));      // index of state bit inside the block
                        dst_bit = 1 << (dst_state % bit_sizeof(ST_BLOCK));
                        src_block = src_state / bit_sizeof(ST_BLOCK);           // index of state block
                        dst_block = dst_state / bit_sizeof(ST_BLOCK);

                        // If transition source is set in current active state vector (divergence happens here)
                        if (src_bit & current_st_vec[src_block]) {
                                // Set transition destination in future active state vector
                                atomicOr(&future_st_vec[dst_block], dst_bit);
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
                        tuple = transition_list[i + wb_transition_start];
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

void run_iNFA(class TransitionGraph *tg, 
    unsigned char **h_input_array, 
    int *input_bytes_array, 
    int array_size,
    int threads_per_block, 
    bool show_match_result,
    bool profiler_mode)
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
    Transition *d_transition_list;                                  // list of transition (source, destination) tuples
    int *d_transition_offset;                                       // index of first transition trigger by each symbol    
    ST_BLOCK *d_init_st_vec, *d_persis_st_vec, *d_final_st_vec;     // state vectors

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
    cudaMalloc((void **)&d_transition_list, sizeof(Transition) * tg->transition_count);
    cudaMalloc((void **)&d_transition_offset, sizeof(int) * (SYMBOL_COUNT + 1));
    cudaMalloc((void **)&d_init_st_vec, sizeof(ST_BLOCK) * vec_len);
    cudaMalloc((void **)&d_persis_st_vec, sizeof(ST_BLOCK) * vec_len); 
    cudaMalloc((void **)&d_final_st_vec, sizeof(ST_BLOCK) * vec_len * array_size); 
    cudaEventRecord(memalloc_end, 0);
    
    // Copy input from host memory into device memory
    cudaEventRecord(memcpy_h2d_start, 0);        
    cudaMemcpy(d_input, h_input, total_input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_offset, h_input_offset, sizeof(int) * (array_size + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transition_list, tg->transition_list, sizeof(Transition) * tg->transition_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_transition_offset, tg->offset_per_symbol, sizeof(int) * (SYMBOL_COUNT + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_init_st_vec, tg->init_states_vector.vector, sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_persis_st_vec, tg->persis_states_vector.vector, sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
    cudaEventRecord(memcpy_h2d_end, 0);

    // Calculate the size of shared memory (for 3 state vectors and transition offset)
    int shem = 3 * vec_len * sizeof(ST_BLOCK) + sizeof(int) * (SYMBOL_COUNT + 1);

    // Launch kernel
    cudaEventRecord(kernel_start, 0);
    nfa_kernel<<<array_size, threads_per_block, shem>>>(d_input,
                                                        d_input_offset,
                                                        d_transition_list,
                                                        d_transition_offset,
                                                        d_init_st_vec,
                                                        d_persis_st_vec,
                                                        d_final_st_vec,
                                                        vec_len);
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
            
            //for (int j = 0; j < accept_rules[i].size(); j++) {
            //        cout << accept_rules[i][j] << " ";
            //}
            //cout << endl;
    } 

    // Free device memory
    cudaEventRecord(memfree_start, 0);
    cudaFree(d_input);
    cudaFree(d_input_offset);
    cudaFree(d_transition_list);
    cudaFree(d_transition_offset);
    cudaFree(d_init_st_vec);
    cudaFree(d_persis_st_vec); 
    cudaFree(d_final_st_vec);
    cudaEventRecord(memfree_end, 0);

    // Free host memory 
    free(h_final_st_vec);
    free(h_input);

    gettimeofday(&end_time, NULL);

    if (show_match_result) show_results(array_size, final_states, accept_rules);

    Profiler(start_time, 
            end_time, 
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
            memfree_end);

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
