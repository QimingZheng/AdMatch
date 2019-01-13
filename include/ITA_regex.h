#ifndef ITA_KERNEL
#define ITA_KERNEL

#include <vector>
#include "src/transition_graph.h"

using namespace std;

struct ita_scratch{
    ita_scratch(ITA_FLAGS flag, char *nfa){
        FLAG_VERIFICATION(flag);
        Kernel_Type kernel;
        if (flag&INFA_KERNEL) kernel=iNFA;
        if (flag&AS_KERNEL) kernel=AS_NFA;
        if (flag&TKO_KERNEL) kernel=TKO_NFA;
        tg = new TransitionGraph(kernel);
        if (!tg->load_nfa_file(nfa)) {
            cerr << "Error: load NFA file " << nfa << endl;
            exit(-1);
        }
    }
    TransitionGraph *tg;
    
    Transition *d_transition_list;                                  // list of transition (source, destination) tuples
    int *d_transition_offset;
    int *d_top_k_offset_per_symbol;
    ST_BLOCK *d_init_st_vec, *d_persis_st_vec, *d_lim_vec;     // state vectors
    ST_BLOCK *d_transition_table;

};

typedef unsigned short int ITA_FLAGS;

#define INFA_KERNEL 1
#define AS_KERNEL (1 << 1)
#define TKO_KERNEL (1 << 2)
#define PROFILER_MODE (1 << 3)
#define SHOW_RESULTS (1 << 4)

void allocScratch(struct ita_scratch &scratch);

void freeScratch(struct ita_scratch &scratch);

void Scan(ITA_FLAGS flag, char *nfa, char *text, vector<int> *accepted_rules);

void BatchedScan(ITA_FLAGS flag, char *nfa, char **text, int *text_len, int str_count, vector<int> *accepted_rules);

#endif