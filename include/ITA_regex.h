#ifndef ITA_KERNEL
#define ITA_KERNEL

#include <vector>

using namespace std;

typedef unsigned short int ITA_FLAGS;

#define INFA_KERNEL 1
#define AS_KERNEL (1 << 1)
#define TKO_KERNEL (1 << 2)
#define PROFILER_MODE (1 << 3)
#define SHOW_RESULTS (1 << 4)

void Scan(ITA_FLAGS flag, char *nfa, char *text, vector<int> *accepted_rules);

void BatchedScan(ITA_FLAGS flag, char *nfa, char **text, int *text_len, int str_count, vector<int> *accepted_rules);

#endif