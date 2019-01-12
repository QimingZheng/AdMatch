#include <iostream>
#include <stdlib.h>
#include <string>
#include "string.h"

#include "include/ITA_regex.h"
#include "src/transition_graph.h"
#include "src/host_functions.h"

using namespace std;

void FLAG_VERIFICATION(ITA_FLAGS flag){
    assert((flag&TKO_KERNEL)+(flag&AS_KERNEL)+(flag&INFA_KERNEL) == 1);
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

    accepted_rules = run_nfa(&tg, h_input_array, input_bytes_array, 1, 1024, 1, flag&PROFILER_MODE);
    for (int i =0; i<accepted_rules[0].size(); i++)
    cout<<accepted_rules[0][i]<<" - ";
    cout<<endl;
}

void BatchedScan(ITA_FLAGS flag, char *nfa, unsigned char *text, int *text_len, int str_count, vector<int> *accepted_rules){
    FLAG_VERIFICATION(flag);
}
