#include <iostream>
#include <stdlib.h>
#include <string>
#include "string.h"

#include "include/ITA_regex.h"
#include "src/transition_graph.h"
#include "src/host_functions.h"

using namespace std;

void FLAG_VERIFICATION(ITA_FLAGS flag){
    assert( ((flag&TKO_KERNEL)+(flag&AS_KERNEL)+(flag&INFA_KERNEL)) == 1);
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
