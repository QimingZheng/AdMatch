#include <iostream>
#include <stdlib.h>
#include <string>
#include "string.h"

#include "include/ITA_regex.h"
#include "src/transition_graph.h"
#include "src/host_functions.h"

using namespace std;

void FLAG_VERIFICATION(ITA_FLAGS flag){
    int a,b,c;
    a = (flag&TKO_KERNEL)>0;
    b = (flag&AS_KERNEL)>0;
    c = (flag&INFA_KERNEL)>0;
    assert(a+b+c==1);
}

void allocScratch(struct ita_scratch &scratch){
    int vec_len = scratch.tg->init_states_vector.block_count;
    if(scratch.tg->kernel==iNFA){
        cudaMalloc((void **)&(scratch.d_transition_list), sizeof(Transition) * scratch.tg->transition_count);
        cudaMalloc((void **)&(scratch.d_transition_offset), sizeof(int) * (SYMBOL_COUNT + 1));
        cudaMalloc((void **)&(scratch.d_init_st_vec), sizeof(ST_BLOCK) * vec_len);
        cudaMalloc((void **)&(scratch.d_persis_st_vec), sizeof(ST_BLOCK) * vec_len); 

        cudaMemcpy(scratch.d_transition_list, scratch.tg->transition_list, sizeof(Transition) * scratch.tg->transition_count, cudaMemcpyHostToDevice);
        cudaMemcpy(scratch.d_transition_offset, scratch.tg->offset_per_symbol, sizeof(int) * (SYMBOL_COUNT + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(scratch.d_init_st_vec, scratch.tg->init_states_vector.vector, sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
        cudaMemcpy(scratch.d_persis_st_vec, scratch.tg->persis_states_vector.vector, sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
    }
    if(scratch.tg->kernel==TKO_NFA){
        cudaMalloc((void **)&(scratch.d_transition_list), sizeof(Transition) * scratch.tg->transition_count);
        cudaMalloc((void **)&(scratch.d_init_st_vec), sizeof(ST_BLOCK) * vec_len);
        cudaMalloc((void **)&(scratch.d_lim_vec), sizeof(ST_BLOCK)*vec_len * SYMBOL_COUNT * TOP_K);
        cudaMalloc((void **)&(scratch.d_top_k_offset_per_symbol), sizeof(int) * SYMBOL_COUNT * TOP_K);

        cudaMemcpy(scratch.d_transition_list, scratch.tg->transition_list, sizeof(Transition) * scratch.tg->transition_count, cudaMemcpyHostToDevice);
        cudaMemcpy(scratch.d_init_st_vec, scratch.tg->init_states_vector.vector, sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
        cudaMemcpy(scratch.d_top_k_offset_per_symbol, scratch.tg->top_k_offset_per_symbol, sizeof(int) * SYMBOL_COUNT * TOP_K, cudaMemcpyHostToDevice);
        for (int i =0;i<SYMBOL_COUNT;i++){
            for(int j = 0; j<TOP_K; j++){
                    cudaMemcpy(&(scratch.d_lim_vec[i*TOP_K*vec_len+j*vec_len]), scratch.tg->lim_vec[i][j].vector, sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
            }
        }
    }
    if(scratch.tg->kernel==AS_NFA){
        cudaMalloc((void **)&(scratch.d_init_st_vec), sizeof(ST_BLOCK) * vec_len);
        cudaMalloc((void **)&(scratch.d_transition_table), sizeof(ST_BLOCK) * vec_len * scratch.tg->state_count * SYMBOL_COUNT);
        cudaMalloc((void **)&(scratch.d_transition_list), sizeof(Transition) * scratch.tg->wb_transition_count);

        cudaMemcpy(scratch.d_init_st_vec, scratch.tg->init_states_vector.vector, sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
        cudaMemcpy(scratch.d_transition_list, scratch.tg->transition_list, sizeof(Transition) * scratch.tg->wb_transition_count, cudaMemcpyHostToDevice);
        for(int i=0;i<SYMBOL_COUNT;i++)
        {
        for(int j=0;j<scratch.tg->state_count;j++)
        {
                cudaMemcpy(&(scratch.d_transition_table[vec_len*(i*scratch.tg->state_count+j)]),
                        scratch.tg->transition_table[i*scratch.tg->state_count+j].vector,
                        sizeof(ST_BLOCK) * vec_len, cudaMemcpyHostToDevice);
        }
        }
    }
}

void freeScratch(struct ita_scratch &scratch){
    if(scratch.tg->kernel==iNFA){
        cudaFree(scratch.d_transition_list);
        cudaFree(scratch.d_transition_offset);
        cudaFree(scratch.d_init_st_vec);
        cudaFree(scratch.d_persis_st_vec);
    }
    if(scratch.tg->kernel==TKO_NFA){
        cudaFree(scratch.d_transition_list);
        cudaFree(scratch.d_lim_vec);
        cudaFree(scratch.d_init_st_vec);
        cudaFree(scratch.d_top_k_offset_per_symbol);
    }
    if(scratch.tg->kernel==AS_NFA){
        cudaFree(scratch.d_transition_list);
        cudaFree(scratch.d_init_st_vec);
        cudaFree(scratch.d_transition_table);
    }
}


void Scan(ITA_FLAGS flag, char *nfa, char *text, vector<int> *accepted_rules){
    FLAG_VERIFICATION(flag);
    Kernel_Type kernel;
    if (flag&INFA_KERNEL) kernel=iNFA;
    if (flag&AS_KERNEL) kernel=AS_NFA;
    if (flag&TKO_KERNEL) kernel=TKO_NFA;

    TransitionGraph tg(kernel);

    if (!tg.load_nfa_file(nfa)) {
        cerr << "Error: load NFA file " << nfa << endl;
        exit(-1);
    }

    unsigned char *h_input_array[1];
    int input_bytes_array[1];
    h_input_array[0]=(unsigned char*)text;
    input_bytes_array[0]=strlen(text);

    run_nfa(&tg, h_input_array, input_bytes_array, 1, 1024, flag&SHOW_RESULTS, flag&PROFILER_MODE, accepted_rules);
}

void BatchedScan(ITA_FLAGS flag, char *nfa, char **text, int *text_len, int str_count, vector<int> *accepted_rules){
    FLAG_VERIFICATION(flag);
    Kernel_Type kernel;
    if (flag&INFA_KERNEL) kernel=iNFA;
    if (flag&AS_KERNEL) kernel=AS_NFA;
    if (flag&TKO_KERNEL) kernel=TKO_NFA;

    TransitionGraph tg(kernel);

    if (!tg.load_nfa_file(nfa)) {
        cerr << "Error: load NFA file " << nfa << endl;
        exit(-1);
    }

    unsigned char *h_input_array[str_count];
    
    for (int i = 0; i < str_count; i++) {
        h_input_array[i] = (unsigned char *) text[i];
    }

    run_nfa(&tg, h_input_array, text_len, str_count, 32, flag&SHOW_RESULTS, flag&PROFILER_MODE, accepted_rules);

}
