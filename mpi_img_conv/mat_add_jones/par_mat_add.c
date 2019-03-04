#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "matrix_generator.h"
#include <string.h>
#include <time.h>

#define ONE_BILLION (double)1000000000.0

int *A_Matrix;
int *B_Matrix;
int *C_Matrix;

typedef struct {
    int rows;
    int cols;
    int start;
    int load;
    int thread;
    int num_threads;
} mystery_box_t;

/**
 * Method for getting the current time
 */
double now(void) {
    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

void *mat_add(void *parameter) {
    mystery_box_t *my_box = (mystery_box_t *)parameter;
    int load = my_box->load;
    int start = my_box->start;
    int rows = my_box->rows;
    int cols = my_box->cols;
    int thread = my_box->thread;
    int size = load * cols;
    for(int i = 0; i < size; i++) {
        //printf("%d: %d %d %d\n", thread, i, load, start);
        *(C_Matrix + (start*thread) + i) = *(A_Matrix + (start*thread) + i) + *(B_Matrix + (start*thread) + i);
    }
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

    int rows = 2048;
    int cols = 2048;
    generate_matrix(rows, cols, "a_parallel.txt");
    generate_matrix(rows, cols, "b_parallel.txt");
    A_Matrix = read_matrix(&rows, &cols, "a_parallel.txt");
    B_Matrix = read_matrix(&rows, &cols, "b_parallel.txt");

    pthread_t threads[num_threads];
    mystery_box_t *boxes[num_threads];
    int size = rows * cols;
    int load_size = size / num_threads;


    for(int i = 0; i < num_threads; i++) {
        int start = load_size * i;
        if((size % num_threads != 0) && (i == num_threads - 1)) {
            load_size = rows - start;
        }
        boxes[i] = malloc(sizeof(mystery_box_t));
        boxes[i]->rows = rows;
        boxes[i]->cols = cols;
        boxes[i]->start = start;
        boxes[i]->load = load_size;
        boxes[i]->thread = i;
        boxes[i]->num_threads = num_threads;
    }

    C_Matrix = malloc(rows * cols * sizeof(int));
    double start_time = now();
    for(int i = 0; i < num_threads; i++) {
        pthread_create(&threads[i], NULL, mat_add, boxes[i]);
    }
    for(int i = 0; i < num_threads; i++) {
        pthread_join( threads[i], NULL);
        free(boxes[i]);
    }

    FILE *fp;
    if((fp = fopen("c_parallel.txt", "w")) == NULL) {
        fprintf(stderr, "Can't open 'c_parallel.txt' for writing\n");
        exit(1);
    }
    fprintf(fp, "%d %d\n", rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            fprintf(fp, " %3d", MAT_ELT(C_Matrix, cols, i, j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("On two %dx%d matrices, matrix addition on %d processors took %5.3f seconds\n", rows, cols, num_threads, now()-start_time);

    return 0;
}
