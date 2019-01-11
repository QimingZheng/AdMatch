#ifndef STATE_VECTOR_H
#define STATE_VECTOR_H

#include <string>
#include <vector>
#include "src/common.h"

using namespace std;

// state bit vectors
class StateVector 
{
public:
        // Functions
        StateVector();
        ~StateVector();
        bool alloc(unsigned int c);     // allocate state vector memory
        void set_bit(unsigned int which);            
        string toString();

        // Variables
        ST_BLOCK *vector;
        unsigned int state_count;               // # of states 
        unsigned int block_count;               // # of state blocks
        unsigned int active_state_count;        // # of active states  
};

// Get active states from a state bit vector
// st_vec:        input state bit vector
// block_len:     # of state blocks the state vector has 
// active_states: store active states
void get_active_states(ST_BLOCK *st_vec, int block_len, vector<ST_T> &active_states);

#endif
