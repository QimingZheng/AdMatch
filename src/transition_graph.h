#ifndef TRANSITION_GRAPH_H
#define TRANSITION_GRAPH_H

#include <assert.h>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

#include "src/common.h"
#include "src/state_vector.h"

using namespace std;

enum Kernel_Type { iNFA, TKO_NFA, AS_NFA };

// Transition source and destination tuple
class Transition {
   public:
    Transition(int a = -1, int b = -1) : src(a), dst(b) {}
    ST_T src;
    ST_T dst;

    friend bool operator<(const Transition& t1, const Transition& t2) {
        if (t1.src < t2.src) {
            return true;
        }

        if (t1.src == t2.src) {
            return t1.dst < t2.dst;
        }

        return false;
    }

    friend bool operator==(const Transition& t1, const Transition& t2) {
        return t1.src == t2.src && t1.dst == t2.dst;
    }
};

// NFA Transition graph
class TransitionGraph {
   public:
    // Functions
    TransitionGraph(Kernel_Type k_t);
    ~TransitionGraph();

    bool load_nfa_file(
        char* file_name);      // construct the NFA transition graph from a file
    bool merge_transitions();  // merge several per-symbol transition tuple
                               // arrays into a global tuple array

    Kernel_Type kernel;

    // Variables
    // General Variables
    vector<Transition>
        transitions_per_symbol[SYMBOL_COUNT];  // transition tuples triggered by
                                               // each symbol
    Transition* transition_list;               // overall transition tuples
    int offset_per_symbol[SYMBOL_COUNT + 1];   // index of first transition
                                               // trigger by each symbol

    vector<ST_T> init_states;    // initial states
    vector<ST_T> persis_states;  // persistent (self-looping) states
    vector<ST_T> accept_states;  // accept states
    unordered_map<ST_T, vector<int> >
        accept_states_rules;  // accept states (key) and corresponding rules
                              // (value)

    StateVector init_states_vector;    // bit vector of initial states
    StateVector persis_states_vector;  // bit vector of persistent states
    StateVector accept_states_vector;  // bit vector of accept states

    int state_count;       // # of states in total
    int transition_count;  // # of transitions in total (exclude transitions of
                           // persistent states)

    // TKO Variables
    int* optimal_k_per_symbol; // [SYMBOL_COUNT + 1]; optimal_k_per_symbol[k+1] - optimal_k_per_symbol[k] = optimal k for symbol[k]
    int* top_k_offset_per_symbol;         //[optimal_k_per_symbol[SYMBOL_COUNT+1]];
    vector<ST_T>* lim_jump_with_offset;  //[optimal_k_per_symbol[SYMBOL_COUNT+1]];
    map<int, set<Transition> > lim_tran_per_symbol_per_offset[SYMBOL_COUNT];
    StateVector* lim_vec;  //[optimal_k_per_symbol[SYMBOL_COUNT+1]];
    int total_transition_count;

    // AS-NFA Variable
    StateVector* transition_table;
    int wb_transition_count;
};

#endif
