#include <stdio.h>
#include <stdbool.h>
#include <assert.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>

#define MAXM (13 * 4) // maximum number of cards
const char RANKS[] = {'3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2'};
const char SUITS[] = {'D', 'C', 'H', 'S'};
int discard[MAXM], dcnt = 0;

int ctoi(char* card) {
    int i = 0, j = 0;
    while (card[0] != SUITS[j]) j++;
    while (card[1] != RANKS[i]) i++;
    return (i << 2) + j;
}

void itoc(int card, char* c) {
    c[0] = SUITS[card & 3]; // mod 4
    c[1] = RANKS[card >> 2]; // div 4
    c[2] = '\0';
}

int input(int cards[]) {
    bool used[MAXM];
    memset(used, 0, sizeof(used));
    
    int m = 0;
    char card[3];
    while (scanf("%s", card) != EOF) {
        int c = ctoi(card);
        if (used[c]) {
            discard[c]++, dcnt++;
            continue;
        }
        used[c] = true;
        cards[m++] = c;
    }

    return m;
}

void rr(const int cards[], int m, int n, int hands[][MAXM], int hands_size[]) {
    for (int i = 0; i < n; i++) hands_size[i] = 0;
    for (int i = 0; i < m; i++) {
        hands[i % n][i / n] = cards[i];
        hands_size[i % n] ++;
    }
}

int find_first(const int hands[][MAXM], const int hands_size[], int n) {
    int li = 0, lval = 999;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < hands_size[i]; j++) {
            if (lval > hands[i][j]) lval = hands[i][j], li = i;
        }
    }
    return li;
}

void dealer(int ptoc[][2], int ctop[][2], int n, int begin) {
    int prev_play = -1, done_cnt = 0, cur = begin, pass_cnt = 0;
    bool winner = 0, dones[n];
    memset(dones, 0, sizeof(dones));

    int cplay, cdone;
    char cplay_str[3];
    while (done_cnt < n - 1) {
        if (!dones[cur]) {
            write(ptoc[cur][1], &prev_play, sizeof(int));
            read(ctop[cur][0], &cplay, sizeof(int));

            if (cplay == -1) {  // child passed
                printf("Parent: child %d passes\n", cur + 1);
                if (++pass_cnt >= n - done_cnt - 1) // everyone has passed
                    pass_cnt = 0, prev_play = -1; // next player can play from smallest
            } else { // child played
                pass_cnt = 0;
                prev_play = cplay;
                itoc(cplay, cplay_str);
                printf("Parent: child %d plays %s\n", cur + 1, cplay_str);
                write(ptoc[cur][1], &prev_play, sizeof(int)); // query if complete?
                read(ctop[cur][0], &cdone, sizeof(int));
                if (cdone) { // child completed
                    done_cnt++, dones[cur] = 1;
                    if (!winner) printf("Parent: child %d is winner\n", cur + 1), winner = 1;
                    else printf("Parent: child %d completes\n", cur + 1);
                    pass_cnt = -1;
                }
            }
        }
        cur = (cur + 1) % n;
        
    }
    int exit_code = -999, loser;
    for (loser = 0; loser < n && dones[loser]; loser++);
    printf("Parent: child %d is loser\n", loser + 1);
    write(ptoc[loser][1], &exit_code, sizeof(int));
}

int find_card(const int hands[], const int hands_size, bool used[], const int prev_play) {
    int res = -1;
    for (int i = 0; i < hands_size; i++) {
        if (used[i]) continue;
        if (hands[i] > prev_play && (res == -1 || hands[res] > hands[i])) res = i;
    }
    if (res != -1) used[res] = 1;
    return res;
}

void player(int id, int ptoc[], int ctop[], const int hands[], const int hands_size) {
    bool used[hands_size];
    memset(used, 0, sizeof(used));
    int buffer, remain = hands_size;
    char play[3];

    read(ptoc[0], &buffer, sizeof(int)); // listen for self-report instruction
    printf("Child %d, pid %d: I have %d cards\n", id + 1, buffer, hands_size);
    printf("Child %d, pid %d:", id + 1, buffer);
    for (int i = 0; i < hands_size; i++) {
        itoc(hands[i], play);
        printf(" %s", play);
    }
    printf("\n");
    write(ctop[1], &buffer, sizeof(int)); // signal self-report complete

    while (remain) {
        read(ptoc[0], &buffer, sizeof(int));
        if (buffer == -999) break; // exit code, game over

        int res = find_card(hands, hands_size, used, buffer);
        if (res == -1) {
            printf("Child %d: pass\n", id + 1);
            write(ctop[1], &res, sizeof(int)); // pass
        } else {
            itoc(hands[res], play);
            printf("Child %d: play %s\n", id + 1, play);
            write(ctop[1], hands + res, sizeof(int));
            used[res] = 1;
            remain--;
            
            int cdone = (remain > 0) ? 0 : 1;
            read(ptoc[0], &buffer, sizeof(int));
            if (cdone) printf("Child %d: I complete\n", id + 1);
            write(ctop[1], &cdone, sizeof(int));
        }
    }
    close(ptoc[0]);
    close(ctop[1]);
    exit(0);
}

void play(const int hands[][MAXM], const int hands_size[], int n, int begin) {
    int id, pid[n], ptoc[n][2], ctop[n][2];
    bool used[MAXM];
    memset(used, 0, sizeof(used));

    for (int i = 0; i < n; i++) {
        pipe(ptoc[i]);
        pipe(ctop[i]);

        pid[i] = fork();
        if (pid[i] == 0) { // child
            id = i;
            close(ptoc[i][1]);
            close(ctop[i][0]);
            break;
        } else { // parent
            id = -1;
            close(ptoc[i][0]);
            close(ctop[i][1]);
        }
    }

    if (id == -1) {
        printf("Parent: the child players are");
        for (int i = 0; i < n; i++) printf(" %d", pid[i]);
        printf("\n");
        if (dcnt > 0) {
            for (int i = 0; i < MAXM; i++) {
                char card[3];
                itoc(i, card);
                while (discard[i]--) printf("Parent: duplicated %s discarded\n", card);
            }
        }
        for (int i = 0; i < n; i++) { // tell child processes to self-report
            int buffer;
            write(ptoc[i][1], pid + i, sizeof(int));
            read(ctop[i][0], &buffer, sizeof(int)); // signals child completed self-report
        }
        dealer(ptoc, ctop, n, begin); // pa
        int status;
        for (int i = 0; i < n; i++) waitpid(pid[i], &status, 0);
        exit(0);
    }
    else player(id, ptoc[id], ctop[id], hands[id], hands_size[id]); // child
}

int main (int argc, char *argv[]) {
    setbuf(stdout, NULL);
    assert(argc == 2);
    int begin, m, n = atoi(argv[1]);
    int cards[MAXM];
    int hands[n][MAXM];
    int hands_size[n];

    m = input(cards);
    rr(cards, m, n, hands, hands_size);
    begin = find_first(hands, hands_size, n);
    play(hands, hands_size, n, begin);
    return 0;
}