#include <iostream>
#include <stdlib.h>
#include <string>
#include "string.h"

#include "ITA_regex.h"

using namespace std;

int main(int argc, char **argv){
    bool result;
    Scan(INFA_KERNEL|PROFILER_MODE, argv[1], argv[2], result);
    return 0;
}