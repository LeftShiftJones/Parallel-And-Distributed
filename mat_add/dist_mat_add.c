#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "matrix_generator.h"

#define MAT_GET(matrix, columns, i, j) *(matrix + (colums * i) + j)
#define MASTER_CORE 0
#define ONE_BILLION (double)1000000000.0
#define DEBUG 0

/**
Mystery Box struct to keep track of important information
*/
typedef struct {
    int rank;           /* Current rank */
    int num_procs;      /* total # processors */
    int proc_load;      /* Size of chunks given to each processor */
    int rows;           /* Rows in A and B */
    int cols;           /* Cosl in A and B */
    int *a_stripe;      /* Partition of A */
    int *b_stripe;      /* Partition of B */
    int *c_stripe;      /* Partition of C */
} mystery_box_t;

/**
Print out matrix values
*/
void mat_print(char *msge, int *a, int m, int n){
    printf("\n== %s ==\n%7s", msge, "");
    for (int j = 0;  j < n;  j++) {
        printf("%6d|", j);
    }
    printf("\n");
    for (int i = 0;  i < m;  i++) {
        printf("%5d|", i);
        for (int j = 0;  j < n;  j++) {
            printf("%7d", MAT_ELT(a, n, i, j));
        }
        printf("\n");
    }
}

/**
 * Method for getting the current time
 */
double now(void) {
    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

/**
Distriute data to all other processors
*/
void distribute_inital_data(mystery_box_t *my_box, int rows, int cols, int num_procs) {
    /* Create and read in matrices to add from */
    generate_matrix(rows, cols, "a.txt");
    generate_matrix(rows, cols, "b.txt");
    int *A = read_matrix(&rows, &cols, "a.txt");
    int *B = read_matrix(&rows, &cols, "b.txt");

    /* number of rows to give to each processor */
    int proc_load = rows / num_procs;
    for(int i = 0; i < num_procs; i++) {
        /* Starting point in the data
            (dependent on processor number) */
        int start = proc_load * i;

        if(i == num_procs-1) {
            if(rows % num_procs != 0) {
                proc_load = rows - start; /* Assign extra rows to last processor */
            }
        }
        /* Assign matrix data for each processor */
        int size = proc_load * cols;
        int *little_a = malloc(size * sizeof(int));
        int *little_b = malloc(size * sizeof(int));
        for(int c = 0; c < size; c++) {
            *(little_a + c) = *(A + start*cols + c);
            *(little_b + c) = *(B + start*cols + c);
        }
        /* Set data for processor 0 */
        if(i==0) {
            my_box->a_stripe = little_a;
            my_box->b_stripe = little_b;
            my_box->proc_load = proc_load;
        }
        /* For all other processors, send data to processor 0 */
        else {
            if(DEBUG) {printf("Distributing data from 0 to processor %d\n", i);}
            MPI_Send(&proc_load, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(little_a, size, MPI_INT, i, 2, MPI_COMM_WORLD);
            MPI_Send(little_b, size, MPI_INT, i, 3, MPI_COMM_WORLD);
        }
    }
}

/**
Receive data for matrices
*/
void receive_inital_data(mystery_box_t *my_box) {
    MPI_Status status;
    int rows        = my_box->rows;
    int cols        = my_box->cols;
    int rank        = my_box->rank;
    int size;
    /* Recieve the number of rows this processor will receive */
    MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    my_box->proc_load = size;
    /* Allocate memory based on number of rows received */
    int *recv_a = malloc(size * cols * sizeof(int));
    int *recv_b = malloc(size * cols * sizeof(int));
    MPI_Recv(recv_a, size * cols, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
    MPI_Recv(recv_b, size * cols, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
    if(DEBUG) {printf("Received initial matrix data on processor %d\n", rank);}
    /* Store matrices to processor's struct */
    my_box->a_stripe = recv_a;
    my_box->b_stripe = recv_b;
}

/**
Adds up the matrix data for each processor
*/
void mat_add (mystery_box_t *my_box) {
    int proc_load   = my_box->proc_load;
    int rows        = my_box->rows;
    int cols        = my_box->cols;
    int size        = proc_load * cols;
    int rank        = my_box->rank;
    int *c          = malloc(size * sizeof(int));
    /* Addition computation */
    for(int i = 0; i < size; i++) {
        *(c + i) = *(my_box->a_stripe + i) + *(my_box->b_stripe + i);
        if(DEBUG) {printf("%d: %d - ", rank, *(c+i));}
    }
    MPI_Barrier(MPI_COMM_WORLD);
    my_box->c_stripe = c;
    /* Send data to 0
        (0's data is now stored in its mystery_box struct) */
    if(rank > 0) {
        MPI_Send(&proc_load, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
        // MPI_Send(c, size, MPI_INT, 0, 5, MPI_COMM_WORLD); //<-- Old method of sending data
        MPI_Request request;
        MPI_Isend(c, size, MPI_INT, 0, 5, MPI_COMM_WORLD, &request);
            if(DEBUG) {printf("Finished calculation on processor %d, sending to 0\n", rank);}
    } else {
        if(DEBUG) {printf("Finished calculation on processor 0\n");}
    }
}

void write_data_to_disk(mystery_box_t *my_box) {
    if(DEBUG) {printf("Beginning write to disk\n");}
    MPI_Status status;
    int rows        = my_box->rows;
    int cols        = my_box->cols;
    int num_procs   = my_box->num_procs;
    int load        = my_box->proc_load;
    FILE *fp;
    if((fp = fopen("c.txt", "w")) == NULL) {
        fprintf(stderr, "Can't open 'c.txt' for writing\n");
        exit(1);
    }
    fprintf(fp, "%d %d\n", rows, cols);
    /* Read 0's matrix addition data */
    for(int i = 0; i < load; i++) {
        for(int j = 0; j < cols; j++) {
            fprintf(fp, " %3d", MAT_ELT(my_box->c_stripe, cols, i, j));
        }
        fprintf(fp, "\n");
    }

    /* Receive and write other processor's data */
    for(int i = 1; i < num_procs; i++) {
        if(DEBUG) {printf("Writing data from processor %d\n", i);}
        int other_size;
        MPI_Recv(&other_size, 1, MPI_INT, i, 4, MPI_COMM_WORLD, &status);
        int *c = malloc(cols * other_size * sizeof(int));
        MPI_Recv(c, cols * other_size, MPI_INT, i, 5, MPI_COMM_WORLD, &status);
        int val; /* Temp variable */
        for(int j = 0; j < other_size; j++) {
            for(int k = 0; k < cols; k++) {
                val = MAT_ELT(c, cols, j, k);
                fprintf(fp, " %3d", val);
            }
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}

int main(int argc, char **argv) {
    int rows = 256;
    int cols = 256;

    /* MPI Elements */
    int num_procs;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rows < num_procs) {
        fprintf(stderr, "Invalid number of processors, exiting...\n");
        exit(1);
    }

    if(!rank && DEBUG) {
        printf("Beginning program...\nProcessor count: %d;\nRow count: %d;\nColumn count: %d\n\n", num_procs, rows, cols);
    }

    /* Set up mystery box */
    mystery_box_t *my_box = malloc(sizeof(mystery_box_t));
    my_box->rows = rows;
    my_box->cols = cols;
    my_box->rank = rank;
    my_box->num_procs = num_procs;
    double start_time = now();
    if(rank == 0) {
        distribute_inital_data(my_box, rows, cols, num_procs);
    } else {
        receive_inital_data(my_box);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    mat_add(my_box);
    MPI_Barrier(MPI_COMM_WORLD);

    if(!rank) {
        write_data_to_disk(my_box);
        printf("On two %dx%d matrices, matrix addition took %5.3f seconds\n", rows, cols, now()-start_time);
    }

    MPI_Finalize();
    if(!rank && DEBUG) {
        printf("\n\nExiting program...\n");
    }
    return 0;
}
