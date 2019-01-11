#include "kernel_common.h"

__device__ bool is_word_char(unsigned char c)
{       
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || (c == '_');
}

void show_results(int array_size, vector<ST_T> *final_states, vector<int> *accept_rules){
    for (int i = 0; i < array_size; i++) {
        cout << "String " << i + 1 << endl;
        
        cout << "Final States : ";
        for (int j = 0; j < final_states[i].size(); j++) {
                cout << final_states[i][j] << " ";
        }
        cout << endl;

        cout << "Accept Rules : ";
        for (int j = 0; j < accept_rules[i].size(); j++) {
                cout << accept_rules[i][j] << " ";
        }                        
        cout << endl;
    }
    cout << endl;
}

void Profiler(struct timeval start_time, 
              struct timeval end_time, 
              int array_size, 
              cudaEvent_t memalloc_start, 
              cudaEvent_t memalloc_end,
              cudaEvent_t memcpy_h2d_start,
              cudaEvent_t memcpy_h2d_end,
              cudaEvent_t kernel_start,
              cudaEvent_t kernel_end,
              cudaEvent_t memcpy_d2h_start,
              cudaEvent_t memcpy_d2h_end,
              cudaEvent_t memfree_start,
              cudaEvent_t memfree_end
){
    float time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    float cuda_time = 0;

    cout << "Total time: " << time << " ms ";
    cout << "(Per string time: " << time / array_size << " ms)" << endl;

    cudaEventElapsedTime(&time, memalloc_start, memalloc_end);
    cout << "Device memory allocation time: " << time << " ms" << endl;
    cuda_time += time;

    cudaEventElapsedTime(&time, memcpy_h2d_start, memcpy_h2d_end);
    cout << "Memory copy host to device time: " << time << " ms" << endl;
    cuda_time += time;

    cudaEventElapsedTime(&time, kernel_start, kernel_end);
    cout << "Kernel execution time: " << time << " ms ";
    cout << "(Per string kernel execution time: " << time / array_size << " ms)" << endl;
    cuda_time += time;

    cudaEventElapsedTime(&time, memcpy_d2h_start, memcpy_d2h_end);
    cout << "Memory copy device to host time: " << time << " ms" << endl;
    cuda_time += time;

    cudaEventElapsedTime(&time, memfree_start, memfree_end);
    cout << "Device memory free time: " << time << " ms" << endl;
    cuda_time += time;

    cout << "CUDA related time: " << cuda_time << " ms" << endl;
}