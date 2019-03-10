#include <stdlib.h>
#include <algorithm>
#include <boost/algorithm/string/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "src/mem_alloc.h"
#include "src/transition_graph.h"

using namespace boost;

bool cmp(const pair<int, set<Transition>> &a,
         const pair<int, set<Transition>> &b) {
    return a.second.size() > b.second.size();
}

vector<pair<int, set<Transition>>> _sort_(map<int, set<Transition>> &_map_) {
    vector<pair<int, set<Transition>>> vec(_map_.begin(), _map_.end());
    sort(vec.begin(), vec.end(), cmp);
    return vec;
}

TransitionGraph::TransitionGraph(Kernel_Type k_t) {
    kernel = k_t;

    // No states and transitions at the beginning
    state_count = 0;
    transition_count = 0;
    transition_list = NULL;
    for (int i = 0; i < SYMBOL_COUNT + 1; i++) {
        offset_per_symbol[i] = 0;
    }

    if (kernel == TKO_NFA) {
        total_transition_count = 0;
        optimal_k_per_symbol = new int[SYMBOL_COUNT+1];
        /*
        top_k_offset_per_symbol = new int[SYMBOL_COUNT * TOP_K];
        lim_jump_with_offset = new vector<ST_T> *[SYMBOL_COUNT];
        lim_vec = new StateVector *[SYMBOL_COUNT];
        for (int i = 0; i < SYMBOL_COUNT; i++) {
            lim_jump_with_offset[i] = new vector<ST_T>[TOP_K];
            lim_vec[i] = new StateVector[TOP_K];
            for (int j = 0; j < TOP_K; j++) {
                lim_jump_with_offset[i][j].clear();
            }
        }
        */
    }
}

TransitionGraph::~TransitionGraph() {
    free_host(transition_list);
    if (kernel == AS_NFA) free_host(transition_table);
    if (kernel == TKO_NFA) {
        delete[] optimal_k_per_symbol;
        delete[] top_k_offset_per_symbol;
        delete[] lim_jump_with_offset;
        delete[] lim_vec;
    }
}

// Construct NFA transition graph from a file
// Return false if the contruction fails
bool TransitionGraph::load_nfa_file(char *file_name) {
    if (kernel == AS_NFA) {
        wb_transition_count = 0;
    }
    ifstream file(file_name);

    // Read # of states from the first transition graph line.
    // Note that we should skip the comment line starts with '#'.
    string line;
    while (true) {
        if (!getline(file, line)) {
            cerr << "Error: read # of states" << endl;
            return false;
        }

        if (line[0] != '#') {
            istringstream iss(line);
            iss >> state_count;

            break;
        }
    }

    // Allocate state bit vectors
    init_states_vector.alloc(state_count);
    persis_states_vector.alloc(state_count);
    accept_states_vector.alloc(state_count);

    if (kernel == TKO_NFA) {
/*
        for (int i = 0; i < SYMBOL_COUNT; i++)
            for (int j = 0; j < TOP_K; j++) {
                lim_vec[i][j].alloc(state_count);
            }
*/
    }

    if (kernel == AS_NFA) {
        if (!alloc_host((void **)&transition_table,
                        state_count * SYMBOL_COUNT * sizeof(StateVector))) {
            return false;
        }
        for (int i = 0; i < state_count * SYMBOL_COUNT; i++)
            transition_table[i].alloc(state_count);
    }

    // Read line by line
    while (getline(file, line)) {
        // We skip the comment line which starts with '#'
        if (line[0] == '#') {
            continue;
        }

        // Remove the '\r' at the end (due to Windows)
        if (line[line.length() - 1] == '\r') {
            line.erase(line.end() - 1);
        }

        // Split the line by ':'
        vector<string> parts;
        split_regex(parts, line, regex("[[:blank:]]*:[[:blank:]]*"));

        // Skip the line whose format is NOT '*:*'
        if (parts.size() != 2) {
            cerr << "Error: cannot parse " << line << endl;
            continue;
        }

        // Split parts[0] by '->' to get the source and destination
        vector<string> src_dst;
        split_regex(src_dst, parts[0], regex("[[:blank:]]*->[[:blank:]]*"));

        // If this is a legal transition with a source and a destination state
        if (src_dst.size() == 2) {
            ST_T src = lexical_cast<ST_T>(src_dst[0]);
            ST_T dst = lexical_cast<ST_T>(src_dst[1]);

            // Split the transition symbol content into multiple transition
            // symbol ranges
            vector<string> atoms;
            split_regex(atoms, parts[1],
                        regex("[[:blank:]]+(?!\\|)[[:blank:]]*"));

            // For each transition symbol range
            for (int i = 0; i < atoms.size(); i++) {
                if (atoms[i].size() == 0) {
                    continue;
                }

                // A transition symbol range has the following two formats:
                // (1) start_symbol|end_symbol or (2) symbol
                vector<string> symbol_range;
                split_regex(symbol_range, atoms[i], regex("\\|"));
                int start, end;

                start = end = lexical_cast<int>(symbol_range[0]);

                if (symbol_range.size() == 2) {
                    end = lexical_cast<int>(symbol_range[1]);
                }

                // We find a self-looping / persis state
                if (src == dst && start == 0 && end == 255) {
                    //if (kernel == iNFA || kernel == TKO_NFA) {
                    if (kernel == iNFA) {
                        persis_states.push_back(src);
                        persis_states_vector.set_bit(src);
                    }

                    if (kernel == AS_NFA) {
                        for (int ss = 0; ss <= 255; ss++)
                            transition_table[ss * state_count + src].set_bit(
                                dst);
                    }

                    if (kernel == TKO_NFA) {
                        for (int t = 0; t < 256; t++) {
                            total_transition_count += 1;
                            if (lim_tran_per_symbol_per_offset[t].count(0) > 0)
                                lim_tran_per_symbol_per_offset[t][0].insert(
                                    Transition(src, dst));
                            else {
                                lim_tran_per_symbol_per_offset[t].insert(
                                    pair<int, set<Transition>>(
                                        0, set<Transition>()));
                                lim_tran_per_symbol_per_offset[t][0].insert(
                                    Transition(src, dst));
                            }
                        }
                    }
                    // We exclude transitions of persis states
                } else {
                    if (kernel == iNFA) {
                        transition_count += (end - start + 1);

                        // Update per-symbol transitions
                        for (int symbol = start; symbol <= end; symbol++) {
                            transitions_per_symbol[symbol].push_back(
                                Transition(src, dst));
                        }
                    }
                    if (kernel == TKO_NFA) {
                        for (int symbol = start; symbol <= end; symbol++) {
                            total_transition_count += 1;
                            if (symbol == WORD_BOUNDARY) {
                                transitions_per_symbol[symbol].push_back(
                                    Transition(src, dst));
                            } else {
                                if (lim_tran_per_symbol_per_offset[symbol]
                                        .count(dst - src) > 0)
                                    lim_tran_per_symbol_per_offset[symbol][dst -
                                                                           src]
                                        .insert(Transition(src, dst));
                                else {
                                    lim_tran_per_symbol_per_offset[symbol]
                                        .insert(pair<int, set<Transition>>(
                                            dst - src, set<Transition>()));
                                    lim_tran_per_symbol_per_offset[symbol][dst -
                                                                           src]
                                        .insert(Transition(src, dst));
                                }
                            }
                        }
                    }
                    if (kernel == AS_NFA) {
                        for (int symbol = start; symbol <= end; symbol++) {
                            if (symbol == WORD_BOUNDARY) {
                                wb_transition_count += 1;
                                transitions_per_symbol[symbol].push_back(
                                    Transition(src, dst));
                            } else
                                transition_table[symbol * state_count + src]
                                    .set_bit(dst);
                        }
                    }
                }
            }

            // Handle an accpeting state and its related rules
        } else if (parts[1].find("accepting") != string::npos) {
            accept_states.push_back(lexical_cast<ST_T>(src_dst[0]));
            accept_states_vector.set_bit(
                lexical_cast<unsigned int>(src_dst[0]));

            // Get related rules
            vector<int> rules;
            vector<string> atoms;
            split_regex(atoms, parts[1], regex("[[:blank:]]+"));

            if (atoms.size() <= 1) {
                cerr << "No accept rules: " << line << endl;
                continue;
            }

            for (int i = 1; i < atoms.size(); i++) {
                if (atoms[i].size() > 0) {
                    rules.push_back(lexical_cast<int>(atoms[i]));
                }
            }

            accept_states_rules.insert(
                pair<ST_T, vector<int>>(lexical_cast<ST_T>(src_dst[0]), rules));

            // Handle an initial state
        } else if (parts[1].find("initial") != string::npos) {
            init_states.push_back(lexical_cast<ST_T>(src_dst[0]));
            init_states_vector.set_bit(lexical_cast<unsigned int>(src_dst[0]));

            // Cannot parse the line
        } else {
            cerr << "Cannot parse: " << line << endl;
        }
    }

    file.close();
    if (kernel == iNFA) merge_transitions();
    if (kernel == TKO_NFA) {
        for (int i = 0; i <= SYMBOL_COUNT; i++) {
            optimal_k_per_symbol[i] = 0;
        }

        vector<pair<int, set<Transition>>> vec[SYMBOL_COUNT];
#pragma omp parallel for
        for (int i = 0; i < SYMBOL_COUNT; i++) {
            vec[i] = _sort_(lim_tran_per_symbol_per_offset[i]);
            for (auto it = vec[i].begin(); it != vec[i].end(); it++) {
                if (it->second.size()<=(state_count/8)/sizeof(ST_T) || optimal_k_per_symbol[i+1] >= MAX_K) {
                    break;
                }
                optimal_k_per_symbol[i+1]+=1;
            }
        }
#pragma omp barrier

    for (int i=1;i<=SYMBOL_COUNT; i++) optimal_k_per_symbol[i] = optimal_k_per_symbol[i-1]+optimal_k_per_symbol[i];
    top_k_offset_per_symbol = new int [optimal_k_per_symbol[SYMBOL_COUNT]];
    lim_jump_with_offset = new vector<ST_T> [optimal_k_per_symbol[SYMBOL_COUNT]];
    lim_vec = new StateVector [optimal_k_per_symbol[SYMBOL_COUNT]];
    for (int i=0; i<optimal_k_per_symbol[SYMBOL_COUNT]; i++) lim_vec[i].alloc(state_count);

#pragma omp parallel for
        for (int i = 0; i < SYMBOL_COUNT; i++) {
            int j = -1;
            for (auto it = vec[i].begin(); it != vec[i].end(); it++) {
                j++;
                if (it->second.size()<=(state_count/8)/sizeof(ST_T) || j >= MAX_K) {
                    for (auto itt = it->second.begin(); itt != it->second.end();
                         itt++) {
                        transitions_per_symbol[i].push_back((*itt));
                        // transition_count += 1;
                    }
                    continue;
                }
                top_k_offset_per_symbol[optimal_k_per_symbol[i] + j] = it->first;
                for (auto itt = it->second.begin(); itt != it->second.end();
                     itt++)
                    lim_jump_with_offset[optimal_k_per_symbol[i] + j].push_back(itt->src);
            }
        }
#pragma omp barrier

#pragma omp parallel for
        for (int i = 0; i < SYMBOL_COUNT; i++) {
            for (int j = optimal_k_per_symbol[i]; j < optimal_k_per_symbol[i+1]; j++) {
                for (int k = 0; k < lim_jump_with_offset[j].size(); k++) {
                    lim_vec[j].set_bit(lim_jump_with_offset[j][k]);
                }
            }
        }
#pragma omp barrier

        for (int i = 0; i < SYMBOL_COUNT; i++)
            transition_count += transitions_per_symbol[i].size();

        merge_transitions();
    }
    if (kernel == AS_NFA) {
        alloc_host((void **)&transition_list,
                   wb_transition_count * sizeof(Transition));
        for (int i = 0; i < transitions_per_symbol[WORD_BOUNDARY].size(); i++)
            transition_list[i] = transitions_per_symbol[WORD_BOUNDARY][i];
    }
    return true;
}

// Merge several per-symbol transition arrays into an array
// Return false if the merge fails
bool TransitionGraph::merge_transitions() {
    // No transitions at all
    if (transition_count <= 0) {
        return true;
    }

    if (!alloc_host((void **)&transition_list,
                    transition_count * sizeof(Transition))) {
        return false;
    }

    int index = 0;
    for (int i = 0; i < SYMBOL_COUNT; i++) {
        // Record the position of the first transition triggered each symbol
        offset_per_symbol[i] = index;

        // Sort transitions
        sort(transitions_per_symbol[i].begin(),
             transitions_per_symbol[i].end());

        // Remove duplicated transitions
        transitions_per_symbol[i].erase(
            unique(transitions_per_symbol[i].begin(),
                   transitions_per_symbol[i].end()),
            transitions_per_symbol[i].end());

        // Copy transitions from per-symbol array into a global array
        for (int k = 0; k < transitions_per_symbol[i].size(); k++) {
            transition_list[index++] = transitions_per_symbol[i][k];
        }
    }

    // End of all transitions (also # of transitions in total)
    offset_per_symbol[SYMBOL_COUNT] = index;

    // For debug
    if (index != transition_count) {
        cerr << "Error: TransitionGraph::merge_transitions()" << endl;
        return false;
    }

    return true;
}
