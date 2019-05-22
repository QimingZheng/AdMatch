# AdMatch

Please Click Star!

## What's AdMatch

Combination of several GPU-regex matching engines, including: **iNFA**, **TKO-iNFA** and **AS-NFA** and **AdMatch**.
*iNFA*, *TKO-iNFA* and *AS-NFA* are three type of gpu-kernel algorithms for regular expression matching task.
AdMatch-regex adaptively switch to the most suitable kernel in each matching step.

## AdMatch Performance

| Engine Type |  Processing Time Per Request (us) | Acceleration-Ratio |
| -- | -- | -- |
| PCRE | 4416 | 1.0X |
| PCRE-jit | 285 | 15.49X |
| RE2 | 462 | 9.56X |
| HyperScan | 127 | 34.77X |
| iNFA | 7258 | 0.61X |
| AdMatch | 62 | 71.23X |

Based on **CRS rule set** & **Bing Search Request**.

## How To Use AdMatch

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
provide regex file, and set appropriate flags, then feed input strings to it, *AdMatch* will do the rest for you.

For examples of using these AdMatch-regex, please turn to tools/sequence.cpp and tools/batch.cpp.

Testing scripts can be found under directory script/.

## Installation Prerequest

- CUDA Library
- nvcc Compiler
- Boost Library
