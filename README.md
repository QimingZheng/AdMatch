# ITA-regex tool

## ITA-regex

ITA means iNFA, TKO-iNFA and AS-NFA, we provide these three type of kernels for users to select.
Two mode is provided:
```
Single string Scanning

Batched Strings Scanning
```
You can use ITA by including ITA_regex.h

For examples of using these ITA-regex, you can turn to tools/sequence.cpp and tools/batch.cpp.

## Build Prerequest

- CUDA
- nvcc
- Boost Library

## TODO

1. Do more **Correctness Verification**
2. Profile more http request
3. Automatically adjust:
    - threads_per_block
    - kernel_selection
4. Flow-Event based handler