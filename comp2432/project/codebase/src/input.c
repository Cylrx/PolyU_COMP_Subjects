//
// input.c
// The Input Module (Part I)
//

#include "utils.h"
#include "input.h"

#include <stdbool.h>

Request fetch_input() {
    char input[1000];
    printf("Please enter booking:\n");
    fgets(input, 1000, stdin); // because gets() is unsafe apparently
    return preprocess_input(input);
}

Request file_input(FILE* file) {
    char input[1000];
    if (fgets(input, 1000, file) != NULL) {
        return preprocess_input(input);
    }
    return (Request){INVALID};
}

Request preprocess_input(char* input) {
    Request req = {INVALID}; // or use {.type = INVALID} in C99 and later.

    strip_no_semicolon(input); // TODO: revise the strip() function here later

    char **result = split(input);

    char tokens[8][100];
    for (int i = 0; i < 8 && result[i]; i++) {
        strncpy(tokens[i], result[i], 99);
        tokens[i][99] = '\0';
        free(result[i]);
    }
    free(result);

    if (compare(tokens[0], "endProgram")) {
        req.type = TERMINATE;
        return req;
    }

    parse_input(tokens, &req);
    return req;
}


void parse_input(const char tokens[8][100], Request* req) {
    const char* type = tokens[0];

    // special requests
    if (compare(type, "addBatch")) {
        // addBatch -xxxxx  
        // e.g. addBatch –batch001.dat 
        req->type = BATCH;
        req->file = malloc(sizeof(char) * 100);
        strcpy(req->file, tokens[1] + 1); // + 1 to skip the '-'
        return;
    }

    if (compare("printBookings", type)) {
        // printBookings –xxx –[fcfs/prio/opti/ALL]
        req->type = PRINT;
        req->algo = malloc(sizeof(char) * 5);
        strcpy(req->algo, tokens[1] + 1); // + 1 to skip the '-'
        return;
    }

    // priority: Event > Reservation > Parking > Essentials
    // use convention: priority value smaller is higher priority
    req->priority = get_priority(type);
    req->type = REQUEST;

    char member = parse_member(tokens[1]); // member = 'A', 'B', 'C' ...
    int start = parse_time(tokens[2], tokens[3]); // YYYY-MM-DD, hh:mm
    int duration = parse_duration(tokens[4]); // n.n

    if (member == 0) {
        req->type = INVALID;
        printf("Invalid Member: %s\n", tokens[1]);
        return;
    }

    if (start == -1) {
        req->type = INVALID;
        printf("Invalid Time: %s %s\n", tokens[2], tokens[3]);
        return;
    }

    if (duration == -1) {
        req->type = INVALID;
        printf("Invalid Duration: %s\n", tokens[4]);
        return; 
    }

    if (start + duration - 1 > parse_time("2025-05-16", "23:59")) {
        req->type = INVALID;
        printf("Invalid Time: %s %s exceeds 2025-05-16\n", tokens[2], tokens[3]);
        return;
    }

    req->member = member;
    req->start = start;
    req->duration = duration;


    if (compare(type, "addParking")) {
        // addParking -aaa YYYY-MM-DD hh:mm n.n bbb ccc; (bbb, ccc are optional "essential items", but must be paired)
        // parking + essentials (optional)
        const char* bbb = tokens[5];
        const char* ccc = tokens[6];

        int essentials_cnt = (bool)(strlen(bbb)) + (bool)(strlen(ccc)); // how many essentials parameter

        if (essentials_cnt && get_valid_pair(bbb) == NULL) {
            req->type = INVALID;
            printf("[addParking] Invalid Essential Item: %s\n", bbb);
            return;
        }

        if (essentials_cnt == 2 && !is_valid_essentials_pair(bbb, ccc)) {
            req->type = INVALID;
            printf("[addParking] Invalid Essentials Pair: %s %s\n", bbb, ccc);
            return;
        }

        req->parking = true;
        req->essential = 0;
        add_essential_value(&(req->essential), bbb);
        add_essential_value(&(req->essential), ccc);
        return;
    }
    
    if (compare(type, "addReservation")) {
        // addReservation -aaa YYYY-MM-DD hh:mm n.n bbb ccc; (bbb, ccc are mandatorily included, must be paired)
        // parking + essentials (mandatory)
        const char* bbb = tokens[5];
        const char* ccc = tokens[6];
        
        int essentials_cnt = (bool)(strlen(bbb)) + (bool)(strlen(ccc));

        if (essentials_cnt != 2) {
            req->type = INVALID;
            printf("[addReservation] Invalid Number of Essentials: Received %d arguments, expected 2\n", essentials_cnt);
            return;
        }

        if (!is_valid_essentials_pair(bbb, ccc)) {
            req->type = INVALID;
            printf("[addReservation] Invalid Essentials Pair: %s %s\n", bbb, ccc);
            return;
        }

        req->parking = true; 
        req->essential = 0;
        add_essential_value(&(req->essential), bbb);
        return;
    }
    
    if (compare(type, "addEvent")) {
        // addEvent -aaa YYYY-MM-DD hh:mm n.n bbb ccc ddd; (bbb, ccc, ddd are optional "essential items", not necessarily paired)
        // parking  + essentials (optional)
    
        for (int i = 5; i <= 7; i++) {
            if (tokens[i][0] && get_valid_pair(tokens[i]) == NULL) {
                req->type = INVALID;
                printf("[addEvent] Invalid Essential Item: %s\n", tokens[i]);
                return;
            }
        }

        const char* bbb = tokens[5];
        const char* ccc = tokens[6];
        const char* ddd = tokens[7];

        req->parking = true;
        req->essential = 0;

        if (bbb[0]) add_essential_value(&req->essential, bbb);
        if (ccc[0]) add_essential_value(&req->essential, ccc);
        if (ddd[0]) add_essential_value(&req->essential, ddd);
        return;
    }
    
    if (compare(type, "bookEssentials")) {
        // bookEssentials -aaa YYYY-MM-DD hh:mm n.n bbb
        // no parking + essentials (mandatory), i.e., only book essentials without parking

        const char* bbb = tokens[5];
        if (strlen(bbb) == 0) {
            req->type = INVALID;
            printf("[bookEssentials] No Essential Item Received!\n");
            return;
        }

        if (get_valid_pair(bbb) == NULL) {
            req->type = INVALID;
            printf("[bookEssentials] Invalid Essential Item: %s\n", bbb);
            return;
        }

        req->parking = false;
        req->essential = 0;
        add_essential_value(&req->essential, bbb);
        return;
    }

    printf("Unrecognized Command: %s\n", type);
    req->type = INVALID;
}

