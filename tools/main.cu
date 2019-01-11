#include <iostream>
#include <stdlib.h>
#include <string>
#include "string.h"

#include "src/transition_graph.h"
#include "src/host_functions.h"

using namespace std;

void usage(char *program)
{
        cerr << "usage: " << program << " nfa_file input_string string_count threads_per_block [-s]" << endl;
        cerr << "               -s: show regex matching result" << endl;
}

int main(int argc, char **argv)
{       
        bool show_match_result;

        if (argc == 5 || (argc == 6 && strcmp(argv[5], "-s") == 0)) {
                show_match_result = (argc == 6);
        } else {
                usage(argv[0]);
                return EXIT_FAILURE;
        }

        int string_count = atoi(argv[3]);
        if (string_count <= 0) {
                cerr << "Error: invalid string_count value " << string_count << endl;
                return EXIT_FAILURE;
        }
   
        int threads_per_block = atoi(argv[4]);
        if (threads_per_block <= 0) {
                cerr << "Error: invalid threads_per_block value " << threads_per_block << endl;
                return EXIT_FAILURE;                
        }

        // Construct the NFA graph
        TransitionGraph tg(iNFA);
        if (!tg.load_nfa_file(argv[1])) {
                cerr << "Error: load NFA file " << argv[1] << endl;
                return EXIT_FAILURE;
        }

        cout << "load NFA file " << argv[1] << endl;
        
        // Run regex matching on GPU
        unsigned char *h_input_array[string_count];     // array of string
        int input_bytes_array[string_count];            // array of string length
        unsigned char *str = (unsigned char*)argv[2];
        int str_len = strlen(argv[2]);

         for (int i = 0; i < string_count; i++) {
                h_input_array[i] = str;
                input_bytes_array[i] = str_len;
        }

        cout << "Launch kernel with " << threads_per_block << " threads per block" << endl;

        run_nfa(&tg, h_input_array, input_bytes_array, string_count, threads_per_block, show_match_result);

        return EXIT_SUCCESS;
}
