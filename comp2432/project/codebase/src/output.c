//
// output.c
// The Output Module (Part III)
//

#include "output.h"

#include "scheduler.h"
#include "state.h"
#include "vector.h"

#include <unistd.h>
#include <stdbool.h>


// In this project, we assume there are only five members.
// See the instruction document, Page 6.
#define MEMBERS_CNT     5

#define DATE_LEN        12
#define START_LEN       8
#define END_LEN         8
#define TYPE_LEN        13
#define DEVICE_LEN      29

#define TOTAL_MINUTES   10080  // 7 * 1440


typedef enum {
    DONE,               // Tell the child to exit / tell the parent the current task is done.
    PRINT_BOOKINGS,     // Tell the child to print its bookings.
    PRINT_REPORT        // Tell the child to print its reports.
} PipeMessageType;


static void print_header() {
    printf("Date         Start    End      Type          Device                        \n");
    printf("===========================================================================\n");
}

static void print_divider() {
    printf("\n... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ...\n\n");
}

static void print_no_record() {
    printf("   No record for this member.\n");
}

static void print_end() {
    printf("   - End -\n\n");
    printf("===========================================================================\n\n");
}


// quite weird here.
char* get_request_typename(const Request* req) {
    switch (req->priority) {
        case 0:
            return "Event";
        case 1:
            return "Reservation";
        case 2:
            return "Parking";
        case 3:
            return "*";
        default:
            return "(Error)";
    }
}


// Adjust the string into a fixed size of `fixed_len`.
// call free() when the adjusted string is out of use!
char* adjust_string(const char* str, size_t fixed_len) {
    char *result = malloc(fixed_len + 1);
    if (result == NULL) {
        perror("failed to allocate memory");
        return NULL;
    }

    size_t len = strlen(str);

    if (len < fixed_len) { // copy and padding
        memcpy(result, str, len);
        for (size_t i = len; i < fixed_len; i++) {
            result[i] = ' ';
        }
    } else { // copy and cut
        memcpy(result, str, fixed_len);
    }

    result[fixed_len] = '\0';
    return result;
}

void get_2_digit_string(char str[], int num) {
    str[0] = (char)('0' + (num / 10));
    str[1] = (char)('0' + (num % 10));
    str[2] = '\0';
}

char* get_date_string(const Request* req) {
    const int total_days = req->start / (24 * 60);
    char day[3];
    get_2_digit_string(day, 10 + total_days);
    char* date_string = malloc(11);
    sprintf(date_string, "2025-05-%s", day);
    return date_string;
}

// Good implementation
char* get_start_string(const Request* req) {
    const int remaining_minutes = req->start % (24 * 60);

    char hour[3], minute[3];
    get_2_digit_string(hour, remaining_minutes / 60);
    get_2_digit_string(minute, remaining_minutes % 60);

    char* start_string = malloc(6);
    sprintf(start_string, "%s:%s", hour, minute);
    return start_string;
}

// Good implementation
char* get_end_string(const Request* req) {
    const int remaining_minutes = (req->start + req->duration) % (24 * 60);

    char hour[3], minute[3];
    get_2_digit_string(hour, remaining_minutes / 60);
    get_2_digit_string(minute, remaining_minutes % 60);

    char* start_string = malloc(6);
    sprintf(start_string, "%s:%s", hour, minute);
    return start_string;
}


void
print_algorithm_report(const char* algo_name, Statistics* stat, const int invalid_cnt) {
    printf(" For %s:\n", algo_name);

    int received_cnt = stat->accepted.size + stat->rejected.size;

    if (received_cnt > 0) {
        printf("         Total Number of Booking Received: %d (100.00%%)\n", received_cnt);

        printf("         Total Number of Booking Assigned: %d (%.2f%%)\n",
            stat->accepted.size,
            (double)stat->accepted.size / received_cnt * 100.0
        );

        printf("         Total Number of Booking Rejected: %d (%.2f%%)\n",
            stat->rejected.size,
            (double)stat->rejected.size / received_cnt* 100.0
        );
    } else {
        printf("         No Bookings are Received Currently.\n");
    }

    printf("\n");


    // Utilization of Time Slot

    double rate_parking = 0.0;
    double rate_battery_cable = 0.0;
    double rate_locker_umbrella = 0.0;
    double rate_inflation_service_valet_parking = 0.0;

    Vector *accepted = &stat->accepted;
    int size = accepted->size;
    for (int i = 0; i < size; i++) {
        Request* req = &accepted->data[i];
        int duration = req->duration;

        if (req->parking) { rate_parking += duration; }
        if (req->essential & 0b100) { rate_battery_cable += duration; }
        if (req->essential & 0b010) { rate_locker_umbrella += duration; }
        if (req->essential & 0b001) { rate_inflation_service_valet_parking += duration; }
    }

    rate_parking /= (TOTAL_MINUTES * 10);
    rate_battery_cable /= (TOTAL_MINUTES * 3);
    rate_locker_umbrella /= (TOTAL_MINUTES * 3);
    rate_inflation_service_valet_parking /= (TOTAL_MINUTES * 3);

    printf("         Utilization of Time Slot:\n");
    printf("               Parking:           - %.2f%%\n", rate_parking * 100.00);
    printf("               Battery:           - %.2f%%\n", rate_battery_cable * 100.00);
    printf("               Cable:             - %.2f%%\n", rate_battery_cable * 100.00);
    printf("               Locker:            - %.2f%%\n", rate_locker_umbrella * 100.00);
    printf("               Umbrella:          - %.2f%%\n", rate_locker_umbrella * 100.00);
    printf("               Inflation Service: - %.2f%%\n", rate_inflation_service_valet_parking * 100.00);
    printf("               Valet Parking:     - %.2f%%\n", rate_inflation_service_valet_parking * 100.00);

    printf("\n");

    printf("         Invalid request(s) made: %d\n", invalid_cnt);

    printf("\n");
}


/**
 * This function prints the requests of each member.
 * @param member_name The name of the member (not the full name, only 'A', 'B', 'C', 'D', or 'E').
 * @param stat_vector The state vector. This can be only stat->accepted or stat->rejected.
 * @param records_cnt The records made by the member.
 * @param print Print the schedule tables or calculate records_cnt only.
 */
void
process_member(const char member_name, const Vector* stat_vector, int* records_cnt, const bool print) {
    for (int j = 0; j < stat_vector->size; j++) {
        const Request* curreq = &stat_vector->data[j];

        if (curreq->member == member_name) {

            if (!print) {
                (*records_cnt)++;
                continue;
            }

            if ((*records_cnt) == 0) {
                print_header();
            }

            (*records_cnt)++;

            char* date_string       = get_date_string(curreq);
            char* start_string      = get_start_string(curreq);
            char* end_string        = get_end_string(curreq);

            char* date_adjusted     = adjust_string(date_string, DATE_LEN);
            char* start_adjusted    = adjust_string(get_start_string(curreq), START_LEN);
            char* end_adjusted      = adjust_string(get_end_string(curreq), END_LEN);
            char* type_adjusted     = adjust_string(get_request_typename(curreq), TYPE_LEN);

            printf("%s ", date_adjusted);
            printf("%s ", start_adjusted);
            printf("%s ", end_adjusted);
            printf("%s ", type_adjusted);

            free(date_adjusted);
            free(start_adjusted);
            free(end_adjusted);
            free(type_adjusted);

            free(date_string);
            free(start_string);
            free(end_string);


            // Print the devices

            bool is_battery_cable                       = curreq->essential & 0b100;
            bool is_locker_umbrella                     = curreq->essential & 0b010;
            bool is_inflationservice_valetparking       = curreq->essential & 0b001;

            int essential_cnt = 0;
            char *essential_names[6];

            if (is_battery_cable) {
                essential_names[essential_cnt++] = "Battery";
                essential_names[essential_cnt++] = "Cable";
            }

            if (is_locker_umbrella) {
                essential_names[essential_cnt++] = "Locker";
                essential_names[essential_cnt++] = "Umbrella";
            }

            if (is_inflationservice_valetparking) {
                essential_names[essential_cnt++] = "Inflation Service";
                essential_names[essential_cnt++] = "Valet Parking";
            }

            if (!essential_cnt) {
                char* asterisk_adjusted = adjust_string("*", DEVICE_LEN);
                printf("%s \n", asterisk_adjusted);
                free(asterisk_adjusted);
            } else {
                char* essential_name_adjusted = adjust_string(essential_names[0], DEVICE_LEN);
                printf("%s \n", essential_name_adjusted);
                free(essential_name_adjusted);

                for (int k = 1; k < essential_cnt; k++) {
                    printf("                                             ");
                    essential_name_adjusted = adjust_string(essential_names[k], DEVICE_LEN);
                    printf("%s \n", essential_name_adjusted);
                    free(essential_name_adjusted);
                }
            }
        }
    }
}


/**
 * This function runs a scheduler and prints the booking information under that scheduling algorithm.
 */
void
schedule_and_print_bookings_single_algo(
    int pipe_ptoc[2], int pipe_ctop[2],
    char* algo_name, Vector* queue, const int invalid_cnt
) {

    // Run the Scheduler

    Tracker* tracker = malloc(sizeof(Tracker));
    Statistics* stat = malloc(sizeof(Statistics));

    init_tracker(tracker);
    init_statistics(stat);

    if (strcmp(algo_name, "FCFS") == 0) {
        run_fcfs(queue, stat, tracker);
    } else if (strcmp(algo_name, "PRIO") == 0) {
        run_prio(queue, stat, tracker);
    } else if (strcmp(algo_name, "OPTI") == 0) {
        run_opti(queue, stat, tracker);
    }

    printf("\n");


    // Print the bookings


    PipeMessageType buffer;

    read(pipe_ptoc[0], &buffer, sizeof(PipeMessageType));


    if (buffer == PRINT_BOOKINGS) {

        printf("*** Parking Booking - ACCEPTED / %s ***\n\n", algo_name);

        for (int i = 0; i < MEMBERS_CNT; i++) {
            const char member_name = (char)('A' + i);
            printf("Member_%c has the following bookings:\n", member_name);

            int records_cnt = 0;
            process_member(member_name, &stat->accepted, &records_cnt, true);

            if (!records_cnt) {
                print_no_record();
            }

            if (i < MEMBERS_CNT - 1) {
                print_divider();
            }
        }


        printf("\n*** Parking Booking - REJECTED / %s ***\n\n", algo_name);

        for (int i = 0; i < MEMBERS_CNT; i++) {
            const char member_name = (char)('A' + i);

            int records_cnt = 0;
            process_member(member_name, &stat->rejected, &records_cnt, false);

            printf("Member_%c (there are %d bookings rejected):\n", member_name, records_cnt);
            records_cnt = 0;

            process_member(member_name, &stat->rejected, &records_cnt, true);

            if (!records_cnt) {
                print_no_record();
            }

            if (i < MEMBERS_CNT - 1) {
                print_divider();
            }
        }

        print_end();

    }

    write(pipe_ctop[1], &(PipeMessageType){DONE}, sizeof(PipeMessageType));

    read(pipe_ptoc[0], &buffer, sizeof(PipeMessageType));

    if (buffer == PRINT_REPORT) {
        print_algorithm_report(algo_name, stat, invalid_cnt);
        write(pipe_ctop[1], &(PipeMessageType){DONE}, sizeof(PipeMessageType));
    }

}

void schedule_and_print_bookings (char *algo, Vector* queue, const int invalid_cnt) {

    bool is_fcfs = strcmp(algo, "fcfs") == 0 || strcmp(algo, "all") == 0 || strcmp(algo, "ALL") == 0;
    bool is_prio = strcmp(algo, "prio") == 0 || strcmp(algo, "all") == 0 || strcmp(algo, "ALL") == 0;
    bool is_opti = strcmp(algo, "opti") == 0 || strcmp(algo, "all") == 0 || strcmp(algo, "ALL") == 0;
    bool is_all = strcmp(algo, "all") == 0 || strcmp(algo, "ALL") == 0;


    if (!is_fcfs && !is_prio && !is_opti && !is_all) {
        printf("Unsupported scheduling algorithm: \"%s\".\n", algo);
    }


    // Build pipes.

    int pipe_ptoc[3][2], pipe_ctop[3][2];

    // FCFS
    if (is_fcfs) {
        pipe(pipe_ptoc[0]);     // FCFS scheduler, parent to child
        pipe(pipe_ctop[0]);     // FCFS scheduler, child to parent

        // child process
        if (fork() == 0) {
            close(pipe_ptoc[0][1]); // No need to write to ptoc.
            close(pipe_ctop[0][0]); // No need to read from ctop.

            schedule_and_print_bookings_single_algo(
                pipe_ptoc[0], pipe_ctop[0],
                "FCFS", queue, invalid_cnt
            );

            close(pipe_ptoc[0][0]);
            close(pipe_ctop[0][1]);

            exit(0);
        }

        // parent process
        close(pipe_ptoc[0][0]);
        close(pipe_ctop[0][1]);
    }

    // PRIO
    if (is_prio) {
        pipe(pipe_ptoc[1]);     // PRIO scheduler, parent to child
        pipe(pipe_ctop[1]);     // PRIO scheduler, child to parent

        // child process
        if (fork() == 0) {
            close(pipe_ptoc[1][1]); // No need to write to ptoc.
            close(pipe_ctop[1][0]); // No need to read from ctop.

            schedule_and_print_bookings_single_algo(
                pipe_ptoc[1], pipe_ctop[1],
                "PRIO", queue, invalid_cnt
            );

            close(pipe_ptoc[1][0]);
            close(pipe_ctop[1][1]);

            exit(0);
        }

        // parent process
        close(pipe_ptoc[1][0]);
        close(pipe_ctop[1][1]);
    }

    // OPTI
    if (is_opti) {
        pipe(pipe_ptoc[2]);     // OPTI scheduler, parent to child
        pipe(pipe_ctop[2]);     // OPTI scheduler, child to parent

        // child process
        if (fork() == 0) {
            close(pipe_ptoc[2][1]); // No need to write to ptoc.
            close(pipe_ctop[2][0]); // No need to read from ctop.

            schedule_and_print_bookings_single_algo(
                pipe_ptoc[2], pipe_ctop[2],
                "OPTI", queue, invalid_cnt
            );

            close(pipe_ptoc[2][0]);
            close(pipe_ctop[2][1]);

            exit(0);
        }

        // parent process
        close(pipe_ptoc[2][0]);
        close(pipe_ctop[2][1]);
    }


    // Tell the children to print the bookings.

    PipeMessageType buffer;

    if (is_fcfs) {
        write(pipe_ptoc[0][1], &(PipeMessageType){PRINT_BOOKINGS}, sizeof(PipeMessageType));
        read(pipe_ctop[0][0], &buffer, sizeof(PipeMessageType));
    }

    if (is_prio) {
        write(pipe_ptoc[1][1], &(PipeMessageType){PRINT_BOOKINGS}, sizeof(PipeMessageType));
        read(pipe_ctop[1][0], &buffer, sizeof(PipeMessageType));
    }

    if (is_opti) {
        write(pipe_ptoc[2][1], &(PipeMessageType){PRINT_BOOKINGS}, sizeof(PipeMessageType));
        read(pipe_ctop[2][0], &buffer, sizeof(PipeMessageType));
    }


    // Print the summary report (if applicable)

    if (is_all) {
        printf("*** Parking Booking Manager - Summary Report ***\n\n");
        printf("Performance:\n\n");

        write(pipe_ptoc[0][1], &(PipeMessageType){PRINT_REPORT}, sizeof(PipeMessageType));
        read(pipe_ctop[0][0], &buffer, sizeof(PipeMessageType));

        write(pipe_ptoc[1][1], &(PipeMessageType){PRINT_REPORT}, sizeof(PipeMessageType));
        read(pipe_ctop[1][0], &buffer, sizeof(PipeMessageType));

        write(pipe_ptoc[2][1], &(PipeMessageType){PRINT_REPORT}, sizeof(PipeMessageType));
        read(pipe_ctop[2][0], &buffer, sizeof(PipeMessageType));

        printf("\n");
    }


    // Close all pipe ends.

    if (is_fcfs) {
        close(pipe_ptoc[0][1]);
        close(pipe_ctop[0][0]);
    }

    if (is_prio) {
        close(pipe_ptoc[1][1]);
        close(pipe_ctop[1][0]);
    }

    if (is_opti) {
        close(pipe_ptoc[2][1]);
        close(pipe_ctop[2][0]);
    }

    wait(NULL);
}
