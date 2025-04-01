//
// utils.h
// The Utilities
//

#ifndef UTILS_H
#define UTILS_H

#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "input.h"
#include "state.h"

// static const char* SEP_CHARS = " \t\n\v\f\r";

// static bool is_separator(const char);

void strip_no_semicolon(char*);

/*
 * Returns the stripped string of the original string
 * Strip means to remove the prefix and suffix spacers. For example, strip("   str  ") = "str".
 */
void strip(char*);

/*
 * Returns the split segments of a string. The split delimiters are " ", \t, \n, \v, \f and \r.
 * For example, split(" hello world lovely-magpie") = {"hello", "world", "lovely-magpie"}.
 * This function is for the parser, where the input string should be stripped.
 * Use strip() to strip the input string before call this function.
 */
char** split(const char*);

bool compare(const char*, const char*);

int parse_time(const char*, const char*);
int parse_duration(const char*);
char parse_member(const char*);
bool is_valid_essentials_pair(const char*, const char*);
const char* get_valid_pair(const char*);
void add_essential_value(char*, const char*);
int get_priority(const char*);

bool try_put(int order, int start, int end, bool parking, char essential, Tracker* tracker);
void try_delete(int order, int start, int end, bool parking, char essential, Tracker* tracker);

#endif //UTILS_H
