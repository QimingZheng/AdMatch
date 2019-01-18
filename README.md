# ITA-regex

## What's ITA-regex

The acronym ITA is made of the initial letters of **i**NFA, **T**KO-iNFA and **A**S-NFA.
*iNFA*, *TKO-iNFA* and *AS-NFA* are three type of gpu-kernel algorithms for regular expression matching task.
ITA-regex provide these three kernel mode for users to select.

## ITA Performance

| Engine Type |  Processing Time Per Request (us) | Acceleration-Ratio |
| -- | -- |
| PCRE | 4416 | 1.0X |
| PCRE-jit | 285 | 15.49X |
| RE2 | 462 | 9.56X |
| HyperScan | 127 | 34.77X |
| iNFA | 7258 | 0.61X |
| ITA | 62 | 71.23X |

Based on **CRS rule set** & **Bing Search Request**.

## How To Use ITA

Interfaces we exposed are listed as follows:

```c++
struct ita_scratch;

void allocScratch(struct ita_scratch &scratch);

void freeScratch(struct ita_scratch &scratch);

void Scan(struct ita_scratch &scratch, char *text, vector<int> *accepted_rules);

void BatchedScan(struct ita_scratch &scratch, char **text, int *text_len,
                 int str_count, vector<int> *accepted_rules);
```

For interpretations for these interfaces, please turn to include/ITA_regex.h, you only need to include ITA_regex.h,
provide regex file, and set appropriate flags, then feed input strings to ITA, ITA will do the rest for you.

For examples of using these ITA-regex, please turn to tools/sequence.cpp and tools/batch.cpp.

## Installation Prerequest

- CUDA Library
- nvcc Compiler
- Boost Library

## Installation Procedures

1. Specify a **BUILD** directory to install, *./* by default.
2. You need to modify the gpu-arch flag in the Makefile according to the computability of your device.
3. make

After installation, you will find the ita library under **BUILD**/lib, and a handy command line tool *bin/batch* under **BUILD**/bin.

## TODO

1. Do more **Correctness Verification**
2. Profile more http request
3. Automatically adjust:
    - threads_per_block
    - kernel_selection
