#include <stdlib.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "string.h"

#include "ITA_regex.h"

namespace bpo = boost::program_options;
using namespace std;

void inputfile_dump(char *filename, char &*input_string){
    int file_size = 0;
    vector<string> strs;
    string str;
    while(getline(cin, str)){
        strs.push_back(str);
        file_size += str.size();
    }
    input_string = new char[file_size];
    for(int i=0; i<strs.size(); i++){
        for (int j=0; j<strs[i].size(); j++){
            input_string.push_back(char(strs[i][j]));
        }
    }
}

int main(int argc, char** argv) {
    bool show_match_result = false, profiler_mode = false;
    string nfa_file, input_string, kernel_type;
    int string_count;
    ITA_FLAGS flag = 0;

    bpo::options_description opt("all options");

    opt.add_options()("nfa,f", bpo::value<string>(&nfa_file)->default_value(""),
                      "nfa_file")(
        "input,i", bpo::value<string>(&input_string)->default_value(""),
        "input_string")("num,n",
                        bpo::value<int>(&string_count)->default_value(0),
                        "string_count")("s,s", "show regex matching result")(
        "p,p", "show profiling result")(
        "k,k", bpo::value<string>(&kernel_type)->default_value("iNFA"),
        "kernel type [iNFA], [TKO], [AS], [AD], iNFA by default")("help,h", "Helper");

    bpo::variables_map vm;

    try {
        bpo::store(parse_command_line(argc, argv, opt), vm);
    } catch (...) {
        std::cout << "Undefined Arguments\n";
        return EXIT_FAILURE;
    }

    bpo::notify(vm);

    if (vm.count("help")) {
        cout << opt << std::endl;
        return EXIT_SUCCESS;
    }

    if (vm.count("s")) show_match_result = 1;
    if (vm.count("p")) profiler_mode = 1;

    flag |= (profiler_mode ? PROFILER_MODE : 0) |
            (show_match_result ? SHOW_RESULTS : 0);

    if (!strcmp(kernel_type.c_str(), "iNFA")) flag |= INFA_KERNEL;
    if (!strcmp(kernel_type.c_str(), "TKO")) flag |= TKO_KERNEL;
    if (!strcmp(kernel_type.c_str(), "AS")) flag |= AS_KERNEL;
    if (!strcmp(kernel_type.c_str(), "AD")) flag |= AD_MATCHER;

    if (string_count <= 0) {
        cerr << "Error: invalid string_count value " << string_count << endl;
        return EXIT_FAILURE;
    }

    // Run regex matching on GPU
    char* h_input_array[string_count];    // array of string
    int input_bytes_array[string_count];  // array of string length

    char* str = NULL;
    inputfile_dump(input_string.c_str(), str);
    char* nfa = strdup(nfa_file.c_str());

    for (int i = 0; i < string_count; i++) {
        h_input_array[i] = str;
        input_bytes_array[i] = strlen(str);
    }

    vector<int> accepted_rules[string_count];

    if(flag & AD_MATCHER) {
        ad_scratch scratch(flag, nfa);
        allocScratch(scratch);
        BatchedScan(scratch, h_input_array, input_bytes_array, string_count,
                    accepted_rules);
        freeScratch(scratch);
    }
    else {
        ita_scratch scratch(flag, nfa);
        allocScratch(scratch);
        BatchedScan(scratch, h_input_array, input_bytes_array, string_count,
                    accepted_rules);
        freeScratch(scratch);
    }

    return EXIT_SUCCESS;
}
