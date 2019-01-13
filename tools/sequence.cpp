#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include "string.h"

#include "ITA_regex.h"

using namespace std;

int main(int argc, char **argv) {
    ita_scratch scratch(INFA_KERNEL | PROFILER_MODE | SHOW_RESULTS, argv[1]);
    allocScratch(scratch);
    vector<int> acc[1];
    Scan(scratch, argv[2], acc);
    freeScratch(scratch);
    return 0;
}