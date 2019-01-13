#ifndef ITA_KERNEL
#define ITA_KERNEL

#include <vector>
#include "src/transition_graph.h"

using namespace std;

typedef unsigned short int ITA_FLAGS;

#define INFA_KERNEL 1
#define AS_KERNEL (1 << 1)
#define TKO_KERNEL (1 << 2)
#define PROFILER_MODE (1 << 3)
#define SHOW_RESULTS (1 << 4)

struct ita_scratch{
    ita_scratch(ITA_FLAGS flags, char *nfa);
    ITA_FLAGS flag;
    TransitionGraph *tg;
    Transition *d_transition_list;                                  // list of transition (source, destination) tuples
    int *d_transition_offset;
    int *d_top_k_offset_per_symbol;
    ST_BLOCK *d_init_st_vec, *d_persis_st_vec, *d_lim_vec;     // state vectors
    ST_BLOCK *d_transition_table;

};

void allocScratch(struct ita_scratch &scratch);

void freeScratch(struct ita_scratch &scratch);

void Scan(struct ita_scratch &scratch, char *text, vector<int> *accepted_rules);

void BatchedScan(struct ita_scratch &scratch, char **text, int *text_len, int str_count, vector<int> *accepted_rules);

#endif