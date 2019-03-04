/**
 * GPU Parallel Assignment
 * @author: Ryan Jones
 * Completed Sunday, December 2nd, 2018
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <limits.h>
#include <time.h>

#define N 1000

/* Reference an element in the TSP distance array. */
#define TSP_ELT(tsp, n, i, j) *(tsp + (i * n) + j)
#define ONE_BILLION (double)1000000000.0

/* Action function for each permuation. */
typedef void (*perm_action_t)(int *v, int n);


/////////////////////////////////////////////////////////////////////
// Kth Perm all the way down...
/////////////////////////////////////////////////////////////////////


typedef struct {
    int *values;  /* Values stored in list */
    int max_size; /* Maximum size allocated */
    int cur_size; /* Size currently in use */
} list_t;

/* Swap v[i] and v[j] */
__device__ void swap(int *v, int i, int j) {
    int t = v[i];
    v[i] = v[j];
    v[j] = t;
}

/* Dump list, including sizes */
__device__ void list_dump(list_t *list) {
    //printf("%2d/%2d", list->cur_size, list->max_size);
    for (int i = 0; i < list->cur_size; i++) {
        printf(" %d", list->values[i]);
    }
    printf("\n");
}

/* Allocate list that can store up to 'max_size' elements */
__device__ list_t *
list_alloc(int max_size) {
    list_t *list = (list_t *)malloc(sizeof(list_t));
    list->values = (int *)malloc(max_size * sizeof(int));
    list->max_size = max_size;
    list->cur_size = 0;
    return list;
}

/* Free a list; call this to avoid leaking memory! */
__device__ void list_free(list_t *list) {
    free(list->values);
    free(list);
}

/* Add a value to the end of the list */
__device__ void list_add(list_t *list, int value) {
    if (list->cur_size >= list->max_size) {
        printf("List full");
        list_dump(list);
        //exit(1);
    }
    list->values[list->cur_size++] = value;
}

/* Return the current size of the list */
__device__ int list_size(list_t *list) {
    return list->cur_size;
}

/* Validate index */
__device__ void _list_check_index(list_t *list, int index) {
    if (index < 0 || index > list->cur_size - 1) {
        printf("Invalid index %d\n", index);
        list_dump(list);
        //exit(1);
    }
}

/* Get the value at given index */
__device__ int list_get(list_t *list, int index) {
    _list_check_index(list, index);
    return list->values[index];
}

/* Remove the value at the given index */
__device__ void list_remove_at(list_t *list, int index) {
    _list_check_index(list, index);
    for (int i = index; i < list->cur_size - 1; i++) {
        list->values[i] = list->values[i + 1];
    }
    list->cur_size--;
}

/* Retrieve a copy of the values as a simple array of integers. The returned
   array is allocated dynamically; the caller must free the space when no
   longer needed.
 */
__device__ int *list_as_array(list_t *list) {
    int *rtn = (int *)malloc(list->max_size * sizeof(int));
    for (int i = 0; i < list->max_size; i++) {
        rtn[i] = list_get(list, i);
    }
    return rtn;
}
 

/* Calculate n! iteratively */
__device__ long factorial(int n) {
    if (n < 1) {
        return 0;
    }

    long rtn = 1;
    for (int i = 1; i <= n; i++) {
        rtn *= i;
    }
    return rtn;
}

long host_factorial(int n) {
    if (n < 1) {
        return 0;
    }

    long rtn = 1;
    for (int i = 1; i <= n; i++) {
        rtn *= i;
    }
    return rtn;
}

/* Return the kth lexographically ordered permuation of an array of size integers
   in the range [0 .. size - 1]. The integers are allocated dynamically and
   should be free'd by the caller when no longer needed.
*/
__device__ int *kth_perm(int k, int size) {
    long remain = k - 1;
    list_t *numbers = list_alloc(size);
    for (int i = 0; i < size; i++) {
        list_add(numbers, i);
    }

    list_t *perm = list_alloc(size);

    for (int i = 1; i < size; i++) {
        long f = factorial(size - i);
        long j = remain / f;
        remain = remain % f;

        list_add(perm, list_get(numbers, j));
        list_remove_at(numbers, j);

        if (remain == 0) {
            break;
        }
    }

    /* Append remaining digits */
    for (int i = 0; i < list_size(numbers); i++) {
        list_add(perm, list_get(numbers, i));
    }

    int *rtn = list_as_array(perm);
    list_free(perm);

    return rtn;
}


/* Given an array of size elements at perm, update the array in place to
   contain the lexographically next permutation. It is originally due to
   Dijkstra. The present version is discussed at:
   http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
 */
__device__ void next_perm(int *perm, int size) {
    int i = size - 1;
    while (perm[i - 1] >= perm[i]) {
        i = i - 1;
    }

    int j = size;
    while (perm[j - 1] <= perm[i - 1]) {
        j = j - 1;
    }

    swap(perm, i - 1, j - 1);

    i++;
    j = size;
    while (i < j) {
        swap(perm, i - 1, j - 1);
        i++;
        j--;
    }
}


/////////////////////////////////////////////////////////////////////
// TSP all the way down...
/////////////////////////////////////////////////////////////////////

/* Create an instance of a symmetric TSP. */
int *create_tsp(int n, int seed) {
    int *tsp = (int *)malloc(n * n * sizeof(int));

    srandom(seed);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            int val = (int)(random() / (RAND_MAX / 100));
            TSP_ELT(tsp, n, i, j) = val;
            TSP_ELT(tsp, n, j, i) = val;
        }
    }
    return tsp;
}

/* Evaluate a single instance of the TSP. */
__device__ int eval_tsp(int *perm, int n, int *distances) {
    /* Calculate the length of the tour for the current permutation. */
    int total = 0;
    for (int i = 0; i < n; i++) {
        int j = (i + 1) % n;
        int from = perm[i];
        int to = perm[j];
        int val = TSP_ELT(distances, n, from, to);
        total += val;
    }
    return total;
}

/* Print a TSP distance matrix. */
void print_tsp(int *tsp, int n, int random_seed) {
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

/////////////////////////////////////////////////////////////////////
// My stuff all the way down...
/////////////////////////////////////////////////////////////////////

/**
 * 
 */
__global__ void compute_shortest_path(int num_cities, int num_threads, int *dists, int *mins) {
    int shortest_length = INT_MAX;
    int tid = threadIdx.x;
    long max_check = factorial(num_cities);
    long tours_to_check = max_check / num_threads;
    int *current = kth_perm((tid+1)*tours_to_check, num_cities);
    int at = tid*tours_to_check;
    do {
        int temp = eval_tsp(current, num_cities, dists);
        if(temp < shortest_length) {
            shortest_length = temp;
        }
        next_perm(current, num_cities);
        at++;
    } while (at < (int)((tid+1)*tours_to_check)-1);
    *(mins + (tid)) = shortest_length;
    free(current); 
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
    fprintf(stderr, "   -n <number of threads>\n");
    exit(1);
}

int main(int argc, char **argv) {
    int random_seed = 42;
    int num_cities = 5;
    int num_threads = 1;
    int ch;
    while ((ch = getopt(argc, argv, "c:hs:n:")) != -1) {
        switch (ch) {
        case 'c':
            num_cities = atoi(optarg);
            break;
        case 's':
            random_seed = atoi(optarg);
            break;
        case 'n':
            num_threads = atoi(optarg);
            break;
        case 'h':
        default:
            usage(argv[0]);
        }
    }
    long fact = host_factorial(num_cities);

    //Checks to see if factorial is essentially less than 1024
    //or the number of specified threads. If so, this statement
    //changes the number of threads to be a 1:1 ratio with the
    //permutations
    if((long)num_threads > fact) {
        num_threads = (int)fact;
        fprintf(stderr, "Too many processors to run effectively...\nRefactoring number of threads to be 1:1...\nNow running on %d GPU threads\n", num_threads);
    }
    
    //create host distances and minimum arrays
    int *h_distances = create_tsp(num_cities, random_seed);
    int *h_min_distances = (int *)malloc(num_threads * sizeof(int));

    //create device distances and minimum arrays
    int *d_distances; 
    int *d_min_distances;
    cudaMalloc((void **)&d_distances, num_cities * num_cities * sizeof(int));
    cudaMalloc((void **)&d_min_distances, num_threads * sizeof(int));

    cudaMemcpy(d_distances, h_distances, num_cities * num_cities * sizeof(int), cudaMemcpyHostToDevice);
    double start_time = now();
    compute_shortest_path<<<1, num_threads>>>(num_cities, num_threads, d_distances, d_min_distances);

    cudaMemcpy(h_min_distances, d_min_distances, num_threads * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_distances);
    cudaFree(d_min_distances);
    
    int lowest = INT_MAX;
    for(int i = 0; i < num_threads; i++) {
        int val = *((h_min_distances) + i);
        if(val < lowest) {
            lowest = val;
        }
    }
    free(h_min_distances);
    free(h_distances);

    printf("Lowest tour %d found on seed %d in %5.3f seconds\n", lowest, random_seed, now() - start_time);
    
    return 0;
}
