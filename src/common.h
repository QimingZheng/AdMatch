#ifndef COMMON_H
#define COMMON_H

#define ST_BLOCK \
    unsigned int  // State bit-block. Each state block has multiple state bits.
#define ST_T unsigned short int  // Type of state.

#define SYMBOL_FETCH unsigned int
#define FETCH_BYTES \
    sizeof(SYMBOL_FETCH)  // # of bytes we fetch from the input string for each
                          // reading

#define SYMBOL_COUNT \
    257  // # of symbols (256 values in 1 byte and word boundary anchor \b)
#define WORD_BOUNDARY 256  // value of word boundary anchor

#define bit_sizeof(a) (sizeof(a) * 8)  // # of bits for a variable / type

#define TOP_K 64 // top-k value for TKO-NFA

#endif