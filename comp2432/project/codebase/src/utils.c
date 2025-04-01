//
// utils.c
// The Utilities
//

#include "utils.h"

#include <assert.h>

/* GENERIC STRING MANIPULATION */

// using isspace() to cover all delimiters (' ', \f, \t, \n, \r, \v, etc.)
void strip_no_semicolon(char* str) {
    if (str == NULL || *str == '\0') return;

    char* start = str;
    while (isspace(*start)) {
        start++;
    }

    char* end = str + strlen(str) - 1;
    while (end > start && isspace(*end) && *end != ';') {
        end--;
    }
    if (*end == ';') end--;

    assert(end >= start);
    size_t len = (unsigned)(end - start + 1);
    memmove(str, start, len);
    str[len] = '\0';
}

char** split(const char* str) {
    char** res = (char**)malloc(sizeof(char*) * 8);
    for (int i = 0; i < 8; i++) {
        res[i] = (char*)malloc(sizeof(char) * 100);
        memset(res[i], 0, sizeof(char) * 100);
    }
    int param_i = 0, j = 0;
    for (int i = 0; str[i]; i++) {
        if (str[i] == ' ') {
            if (param_i >= 8) break;
            else param_i++, j = 0;
            continue;
        }
        res[param_i][j++] = str[i];
    }
    return res;
}

[[deprecated("Use strip_no_semicolon() instead")]]
void strip(char* str) {
    int n = (int)strlen(str);
    printf("stripping \"%s\"\n", str);
    int i = 0, pre = 0;
    while (str[pre] == ' ') pre++;
    for (i = pre; i < n && str[i] != ';'; i++) {
        str[i - pre] = str[i];
    }
    str[i - pre] = '\0';
}

bool compare(const char* str1, const char* str2) {
    for (int i = 0; str1[i] == str2[i]; i++) {
        if (str1[i] == 0) return true;
    }
    return false;
}


/* INPUT PARSING */

int parse_time(const char* date, const char* time) {
    // date format: YYYY-MM-DD
    // time format: hh:mm
    // return the time in minutes
    // error handling: return -1 if 
    //      1. the date         const char ccc[] = tokens[6];
    //      2. wrong format, 
    //      3. or not within range of 2025-05-10 ~ 2025-05-16

    int year, month, day, hour, minute;
    
    if (sscanf(date, "%d-%d-%d", &year, &month, &day) != 3 ||
        sscanf(time, "%d:%d", &hour, &minute) != 2)
        return -1;
    
    if (year != 2025 || month != 5 || day < 10 || day > 16 ||
        hour < 0 || hour > 23 || minute < 0 || minute > 59)
        return -1;
    
    return ((day - 10) * 24 + hour) * 60 + minute;
}

int parse_duration(const char* duration) {
    // duration format: n.n
    // return the duration in minutes
    // error handling: return -1 if
    //      1. the duration is invalid
    //      2. wrong format

    float hours;
    if (sscanf(duration, "%f", &hours) != 1) return -1;
    if (hours <= 0) return -1;
    return (int)(hours * 60);
}

char parse_member(const char* member) {
    // member format: '-member_A', '-member_B', '-member_C', '-member_D', '-member_E'
    // return the member 'A', 'B', 'C' ... in char
    // error handling: return 0 if
    //      1. member other than A, B, C, D, E.
    //      2. wrong format (Expected format: "memberX" where X is A-E)

    if (strlen(member) != 9 || strncmp(member, "-member_", 8) != 0) return 0;
    char member_char = member[8];
    if (member_char < 'A' || member_char > 'E' || member[9] != '\0') return 0;
    return member_char;

}

bool is_valid_essentials_pair(const char* bbb, const char* ccc) {
    // bbb, ccc are essential items
    // return true if they are valid paired, false otherwise
    // valid pairs are: 
    // [battery]+[cables] or [locker]+[umbrella] or [InflationService] + [valetPark]

    const char* valid_pair = get_valid_pair(bbb);
    if (valid_pair == NULL) return false;
    return compare(valid_pair, ccc);
}

const char* get_valid_pair(const char* essential) {
    // essential is one of the essential items
    // return the paired essential item
    // return NULL if the essential item is invalid

    if (compare(essential, "battery")) return "cable";
    if (compare(essential, "cable")) return "battery";
    if (compare(essential, "locker")) return "umbrella";
    if (compare(essential, "umbrella")) return "locker";
    if (compare(essential, "InflationService")) return "valetPark";
    if (compare(essential, "valetPark")) return "InflationService";
    return NULL;
}

void add_essential_value(char* original_code, const char* essential) {
    // struct Request object uses binary to represent the requested essential items
    // It uses 3 bit binary, each bit represent [battery + cable], [locker + umbrella], [inflation service + valet parking] respectively
    // E.g., 0b011 = (battery + cable) + (inflation + valet);
    // this function updates the binary code based on the given

    if (compare(essential, "battery") || compare(essential, "cable")) *original_code |= 0b100;
    if (compare(essential, "locker") || compare(essential, "umbrella")) *original_code |= 0b010;
    if (compare(essential, "InflationService") || compare(essential, "valetPark")) *original_code |= 0b001;
}

int get_priority(const char* type) {
    // priority: Event > Reservation > Parking > Essentials
    // use convention: priority value smaller is higher priority
    // return the priority value based on the given type

    if (compare(type, "addEvent")) return 0;
    if (compare(type, "addReservation")) return 1;
    if (compare(type, "addParking")) return 2;
    if (compare(type, "bookEssentials")) return 3;
    return 4;
}

// Try to response a request.
// This function will process both parking request and essential request(s).
bool try_put(int order, int start, int end, bool parking, char essential, Tracker* tracker) {
    assert(order > 0);
    unsigned pk = 999, ek[3];
    ek[0] = ek[1] = ek[2] = 999;

	/* TRY PARKING */

	if (parking) {
		int buffer[10];
		segtree_range_query(tracker->park, start, end, buffer);
		for (unsigned k = 0; k < 10; k++) {
			if (buffer[k] == 0) {
                pk = k;
                break;
            }
            if (k == 9) return false;
		}
	}

    SegTree* st_list[3] = {
        tracker->bc,
        tracker->lu,
        tracker->vi
    };

    if (essential > 0) {
        for (int e = 0; e < 3; e++) {
            if (essential & (1 << (2 - e))) {
                int buffer[3];
                segtree_range_query(st_list[e], start, end, buffer);
                for (unsigned k = 0; k < 3; k++) {
                    if (buffer[k] == 0) {
                        ek[e] = k;
                        break;
                    }
                    if (k == 2) return false;
                }
            }
        }
    }

    if (pk != 999) {
        segtree_range_set(tracker->park, pk, start, end, order);
    }

    for (int i = 0; i < 3; i++) {
        if (ek[i] != 999) {
            segtree_range_set(st_list[i], ek[i], start, end, order);
        }
    }

    return true;
}

void try_delete(int order, int start, int end, bool parking, char essential, Tracker* tracker) {
    assert(order > 0);
    if (parking) {
        int buffer[10];
        segtree_range_query(tracker->park, start, end, buffer);
        for (unsigned k = 0; k < 10; k++) {
            if (buffer[k] == order) {
                segtree_range_set(tracker->park, k, start, end, 0);
                break;
            }
            assert(k != 9);
        }
    }

    if (essential > 0) {
        SegTree* st_list[3] = {
            tracker->bc,
            tracker->lu,
            tracker->vi
        };
        for (int e = 0; e < 3; e++) {
            if (essential & (1 << (2 - e))) {
                int buffer[3];
                segtree_range_query(st_list[e], start, end, buffer);
                for (unsigned k = 0; k < 3; k++) {
                    if (buffer[k] == order) {
                        segtree_range_set(st_list[e], k, start, end, 0);
                        break;
                    }
                    assert(k != 2);
                }
            }
        }
    }
}