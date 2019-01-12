#include <iostream>
#include <vector>
#include <stdlib.h>
#include <string>
#include "string.h"

#include "ITA_regex.h"

using namespace std;

int main(int argc, char **argv){
    vector<int> acc[1];
    Scan(INFA_KERNEL|PROFILER_MODE|SHOW_RESULTS, argv[1], argv[2], acc);
    return 0;
}