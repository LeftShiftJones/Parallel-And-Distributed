#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include "matrix_generator.h"
#include <string.h>
#include <time.h>

int *A_Matrix;
int *B_Matrix;
int *C_matrix;

typedef struct {
    int start;
    int load;
    int thread;
    int num_threads;
} mystery_box_t;

void *mat_add(void *parameter) {

}

int main(int argc, char **argv) {
    int num_threads = 1;
    int ch;
    while ((ch = getopt(argc, argv, "n:")) != -1)
    {
        switch (ch)
        {
        case 'n':
            num_threads = atoi(optarg);
            if(num_threads < 1) {
                fprintf(stderr, "Invalid processor count, exiting...\n");
                exit(1);
            }
            break;
        default:
            fprintf(stderr, "Wrong runtime arguments, exiting...\n");
            exit(1);
        }
    }

    int rows = 256;
    int cols = 256;

    pthread_t threads[num_threads];
    mystery_box_t *boxes[num_threads];

    for(int i = 0; i < num_threads; i++) {
        boxes[i] = malloc(sizeof(mystery_box_t));
        boxes[i]->
    }

    return 0;
}
