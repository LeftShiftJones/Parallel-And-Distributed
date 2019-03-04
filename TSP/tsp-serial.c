#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>

/* Original permuation code due to D. Jimenez, UT Austin
 * http://faculty.cse.tamu.edu/djimenez/ut/utsa/cs3343/
 */

/* Requires C99 compiler (gcc: -std=c99) */
#define DEBUG 0
#define debug_printf(fmt, ...)                 \
    do {                                       \
        if (DEBUG)                             \
            fprintf(stderr, fmt, __VA_ARGS__); \
    } while (0)

/* Action function for each permuation. */
typedef void (*perm_action_t)(int *v, int n);

/* Reference an element in the TSP distance array. */
#define TSP_ELT(tsp, n, i, j) *(tsp + (i * n) + j)
#define ONE_BILLION (double)1000000000.0


/* Swap array elements. */
void swap(int *v, int i, int j) {
    int t = v[i];
    v[i] = v[j];
    v[j] = t;
}

/* Helper function to compute permutations recursively. */
void perms(int *v, int n, int i, perm_action_t action) {
    int j;

    if (i == n) {
        /* At the end of the array, we have a permutation we can use. */
        action(v, n);
    }
    else {
        /* Recursively explore the permutations from index i to (n - 1). */
        for (j = i; j < n; j++) {
            /* Array with i and j switched */
            swap(v, i, j);
            perms(v, n, i + 1, action);
            /* Swap back to the way they were */
            swap(v, i, j);
        }
    }
}

/* Generate permutations of the elements i to (n - 1). */
void permutations(int *v, int n, perm_action_t action) {
    perms(v, n, 0, action);
}

/* Trivial action to pass to permutations--print out each one. */
void print_perm(int *perm, int n, char *msge) {
    for (int j = 0; j < n; j++) {
        printf("%2d ", perm[j]);
    }
    printf(" - %s\n", msge);
}

/* No-op action */
void nop(int *v, int n) {
    return;
}

int num_cities = 5;
int shortest_length = INT_MAX;
int num_as_short = -1;
int num_trials = 0;
int random_seed = 42;

/* Create an instance of a symmetric TSP. */
int *create_tsp(int n) {
    int *tsp = malloc(n * n * sizeof(int));

    srandom(random_seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            int val = (int)(random() / (RAND_MAX / 100));
            TSP_ELT(tsp, n, i, j) = val;
            TSP_ELT(tsp, n, j, i) = val;
        }
    }
    return tsp;
}

/* Print a TSP distance matrix. */
void print_tsp(int *tsp, int n) {
    printf("TSP (%d cities - seed %d)\n    ", n, random_seed);
    for (int j = 0; j < n; j++) {
        printf("%3d|", j);
    }
    printf("\n");
    for (int i = 0; i < n; i++) {
        printf("%2d|", i);
        for (int j = 0; j < n; j++) {
            printf("%4d", TSP_ELT(tsp, n, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

/* Evaluate a single instance of the TSP. */
void eval_tsp(int *perm, int n) {
    /* Initialize the distances array once per program run. */
    static int *distances = NULL;
    if (NULL == distances) {
        distances = create_tsp(num_cities);
        print_tsp(distances, num_cities);
    }

    /* Calculate the length of the tour for the current permutation. */
    int total = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        int from = perm[i];
        int to = perm[j];
        int val = TSP_ELT(distances, n, from, to);
        debug_printf("tsp[%d, %d] = %d\n", from, to, val);
        total += val;
    }

#if DEBUG
    print_perm(perm, n, "PERM");
#endif

    /* Gather statistics. */
    if (total <= shortest_length) {
        char buf[80];
        sprintf(buf, "len %4d - trial %12d", total, num_trials);
        print_perm(perm, n, buf);

        if (total == shortest_length) {
            num_as_short++;
        }
        else
        {
            num_as_short = 1;
        }
        shortest_length = total;
    }
    num_trials++;
    debug_printf("Total %d\n", total);
}


//get current time
double now(void) {
    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

void usage(char *prog_name) {
    fprintf(stderr, "usage: %s [flags]\n", prog_name);
    fprintf(stderr, "   -h\n");
    fprintf(stderr, "   -c <number of cities>\n");
    fprintf(stderr, "   -s <random seed>\n");
    exit(1);
}

int main(int argc, char **argv) {
    /* Use "random" random seed by default. */
    random_seed = time(NULL);

    int ch;
    while ((ch = getopt(argc, argv, "c:hs:")) != -1) {
        switch (ch) {
        case 'c':
            num_cities = atoi(optarg);
            break;
        case 's':
            random_seed = atoi(optarg);
            break;
        case 'h':
        default:
            usage(argv[0]);
        }
    }

    /* Initialize permutation of cities. */
    int order[num_cities];
    for (int i = 0; i < num_cities; i++) {
        order[i] = i;
    }
    double start_time = now();
    /* "Travel, salesman!" */
    permutations(order, num_cities, eval_tsp);

    /* Report. */
    printf("\n");
    printf("Trials %d\n", num_trials);
    float percent_as_short = (float)num_as_short / (float)num_trials * 100.0;
    printf("Shortest %d - %d tours - %.6f%%\n",
           shortest_length, num_as_short, percent_as_short);
    printf("Program took %5.3f seconds to run.", now() - start_time);
    printf("\n");
}
