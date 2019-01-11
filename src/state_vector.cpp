#include <iostream>
#include <sstream>
#include "string.h"
#include "src/state_vector.h"
#include "src/mem_alloc.h"

StateVector::StateVector()
{
        state_count = 0;
        block_count = 0;
        active_state_count = 0;

        vector = NULL;                
}

StateVector::~StateVector()
{
        //cout << "~StateVector()" << endl;
        if (vector) {
                free_host(vector);
        }
}

// Allocate memory for 'c' states
// Return true if the allocation succeeds. 
bool StateVector::alloc(unsigned int c)   
{
        // Free memory
        if (vector) {
                free_host(vector);
                state_count = block_count = active_state_count = 0;
        }

        state_count = c;
        // Ceil of interger devision state_count / bit_sizeof(ST_BLOCK)
        block_count = (state_count + bit_sizeof(ST_BLOCK) - 1) / bit_sizeof(ST_BLOCK);

        /*cout << "state count: " << state_count << endl;
        cout << "state block count: " << block_count << endl;*/

        // If no enough memory
        if (!alloc_host((void**)&vector, block_count * sizeof(ST_BLOCK))) {
                state_count = block_count = active_state_count = 0;
                return false;
        }

        // Set all state bits to 0
        memset(vector, 0, block_count * sizeof(ST_BLOCK));

        return true;
}

// Set 'which' state bit to 1
void StateVector::set_bit(unsigned int which)
{
        if (which >= state_count)
                return;

        // Get the state block index
        unsigned int block_index = which / bit_sizeof(ST_BLOCK);
        // Get the offset inside the state block
        unsigned int offset = which % bit_sizeof(ST_BLOCK);

        vector[block_index] |= (1 << offset);
        active_state_count++;
}

// State vector to string
string StateVector::toString()
{
        stringstream ss;

        unsigned char *ptr = (unsigned char*)vector;
        // Get # of bytes of 'vector'
        unsigned int byte_count = block_count * sizeof(ST_BLOCK);
        // # of state bits that we have gone through
        unsigned int states = 0;
        unsigned int active_states = 0;

        ss << "{" << active_state_count << "/" << state_count << "} ";

        // For each byte
        for (int i = 0; i < byte_count; i++) {
                // For each bit inside the byte (low to high)
                for (unsigned char mask = 0x01; mask != 0x00 && states < state_count; mask <<= 1) {
                        if (int(ptr[i] & mask) != 0) {    
                                ss << 1;
                                active_states++;
                        } else {                
                                ss << 0;
                        }
                        
                        states++;
                        if (states % 4 == 0) {
                                ss << " ";
                        }
                }
        }

        // For debug
        if (active_states != active_state_count) {
                cerr << "Error: StateVector::toString()" << endl;
        }

        return ss.str();        
}

// Get active states from a state bit vector
// st_vec:        input state bit vector
// block_len:     # of state blocks the state vector has 
// active_states: store active states
void get_active_states(ST_BLOCK *st_vec, int block_len, vector<ST_T> &active_states)
{
        // For each block
        for (int block = 0; block < block_len; block++) {
                ST_BLOCK block_val = st_vec[block];
                int st_index = block * bit_sizeof(ST_BLOCK);

                // For each bit inside the block
                while (block_val) {
                        // bit = 1
                        if ((block_val & 1) != 0) {
                                active_states.push_back(st_index);
                        }

                        block_val = block_val >> 1;
                        st_index++;
                }
        } 
}
