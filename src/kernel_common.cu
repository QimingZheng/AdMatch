#include <iomanip>
#include "kernel_common.h"

__device__ bool is_word_char(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9') || (c == '_');
}

__device__ int warpReduceSum(int val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down(val, offset);
  return val;
}

__device__ int blockReduceSum(int val) {
  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) shared[0] = warpReduceSum(val); //Final reduce within first warp

  __syncthreads();

  return shared[0];
}

void show_results(int array_size, vector<ST_T> *final_states,
                  vector<int> *accept_rules) {
    for (int i = 0; i < array_size; i++) {
        cout << "==================\n";
        cout << "String [" << i + 1 << "] :\n";

        cout << "Final States : ";
        for (int j = 0; j < final_states[i].size(); j++) {
            cout << final_states[i][j] << " ";
        }
        cout << endl;

        cout << "Accept Rules : ";
        for (int j = 0; j < accept_rules[i].size(); j++) {
            cout << accept_rules[i][j] << " ";
        }
        cout << "\n==================\n";
        cout << endl;
    }
}

void Profiler(struct timeval start_time, struct timeval end_time,
              int array_size, cudaEvent_t memalloc_start,
              cudaEvent_t memalloc_end, cudaEvent_t memcpy_h2d_start,
              cudaEvent_t memcpy_h2d_end, cudaEvent_t kernel_start,
              cudaEvent_t kernel_end, cudaEvent_t memcpy_d2h_start,
              cudaEvent_t memcpy_d2h_end, cudaEvent_t memfree_start,
              cudaEvent_t memfree_end) {
    float time = (end_time.tv_sec - start_time.tv_sec) * 1000.0 +
                 (end_time.tv_usec - start_time.tv_usec) / 1000.0;
    float cuda_time = 0;

    cout << setw(20) << "Total: " << setw(10) << time << " ms\n";
    cout << setw(20) << "Per-string: " << setw(10) << time / array_size
         << " ms\n";

    cudaEventElapsedTime(&time, memalloc_start, memalloc_end);
    cout << setw(20) << "Device MemAlloc: " << setw(10) << time << " ms\n";
    cuda_time += time;

    cudaEventElapsedTime(&time, memcpy_h2d_start, memcpy_h2d_end);
    cout << setw(20) << "MemCpy H2D: " << setw(10) << time << " ms\n";
    cuda_time += time;

    cudaEventElapsedTime(&time, kernel_start, kernel_end);
    cout << setw(20) << "Kernel Exec: " << setw(10) << time << " ms\n";
    cout << setw(20) << "Per-string Kernel: " << setw(10) << time / array_size
         << " ms\n";
    cuda_time += time;

    cudaEventElapsedTime(&time, memcpy_d2h_start, memcpy_d2h_end);
    cout << setw(20) << "MemCpy D2H: " << setw(10) << time << " ms\n";
    cuda_time += time;

    cudaEventElapsedTime(&time, memfree_start, memfree_end);
    cout << setw(20) << "Device MemFree: " << setw(10) << time << " ms\n";
    cuda_time += time;

    cout << setw(20) << "CUDA time: " << setw(10) << cuda_time << " ms\n";
}
