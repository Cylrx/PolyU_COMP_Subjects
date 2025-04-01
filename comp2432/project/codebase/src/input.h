//
// input.h
// The Input Module (Part I)
// The input module includes the console ad the parser.
// To make the program structure well, the console and parser are combined into this header.
//

#ifndef INPUT_H
#define INPUT_H

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#define BUFFER_SIZE 512

typedef struct Time {
    int y, m, d;
    int hour, min;
} Time;

typedef enum RequestType {
    INVALID,    // undefined or invalid
    TERMINATE,  // endProgram
    PRINT,      // printBookings
    BATCH,      // addBatch
    REQUEST     // addReservation, addEvent, bookEssentials, etc.
} RequestType;


/**
 * Struct type for storing various information of a request.
 *
 * @param type          The type of the request (PRINT, BATCH, NORMAL, INVALID)
 * @param member        The member who made the request (A, B, C, ...)
 * @param start         The starting time of the request (int, in minutes)
 * @param duration      The duration of the request (int, in minutes)
 * @param priority      The priority of the request (int, smaller is higher)
 * @param parking       Whether the request needs parking
 * @param essential     The essential items requested (3-bit binary)
 * @param file          The file name of the batch request (only valid for BATCH)
 * @param algo          The algorithm name for printing (only valid for PRINT)
 * 
 */
typedef struct Request {
    RequestType type;
    char member; // 'A', 'B', 'C' ... 
    int start; 
    int duration;
    int priority; 
    int order; // start from 1
    bool parking; // need parking or not

    // 3 bit binary, battery + cable, locker + umbrella, InflationService + valetPark respectively
    // E.g., 0b011 = (battery + cable) + (InflationService + valetPark);
    char essential;
    char* file;
    char* algo;
} Request;

Request fetch_input();
Request file_input(FILE* file);
Request preprocess_input(char*);

void parse_input(const char tokens[8][100], Request* req);


#endif //INPUT_H
