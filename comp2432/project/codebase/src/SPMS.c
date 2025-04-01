//
// COMP2432 Operating Systems Group Project
// SPMS - Smart Parking Management System
// Project Pica Magpie [https://en.wikipedia.org/wiki/Magpie]
//
// WANG Yuqi
// YANG Xikun
// LIU Siyuan
// JIN Yixiao
//

#include "input.h"
#include "output.h"
#include "scheduler.h"
#include "state.h"
#include "rng.h"
#include "utils.h"

#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>


int main() {
    int invalid_cnt = 0;
    seed(1389ULL, 74135ULL);
    

    Vector* queue = malloc(sizeof(Vector)); 
    vector_init(queue);

    printf("~~ WELCOME TO PolyU ~~\n");

    while (true) {
        Request req = fetch_input();
        
        switch (req.type) {
            case BATCH: {
                bool is_end = process_batch(queue, &req, &invalid_cnt);
                printf("-> [Pending]\n")
                if (!is_end) {
                    break;
                }
                __attribute__((fallthrough));
            }
            case TERMINATE:
                printf("Bye!");
                return 0;
            case PRINT: {
                const pid_t pid = fork();
                if (pid < 0) {
                    perror("fork");
                    exit(0);
                } else if (pid == 0) {
                    schedule_and_print_bookings(req.algo, queue, invalid_cnt);
                    exit(0);
                } else {
                    wait(NULL);
                    printf("-> [Done]\n");
                }
                break;
            }
            case REQUEST:
                process_request(queue, &req);
                // Scheduling algorithms will be called in case PRINT. [Revision Mar 25]
                printf("-> [Pending]\n")
                break;
            case INVALID:
                printf("-> [Invalid]\n")
                invalid_cnt++;
                break;
            default:
                break;
        }
    }

    return 0;
}
