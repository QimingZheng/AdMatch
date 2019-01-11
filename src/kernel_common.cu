#include "kernel_common.h"

__device__ bool is_word_char(unsigned char c)
{       
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || (c == '_');
}