#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <string.h>
#include "matrix_generator.h"

#define MAT_GET(matrix, columns, i, j) *(matrix + (colums * i) + j)
#define MASTER_CORE 0
#define DEBUG 1

/**
Mystery Box struct to keep track of important information
*/
typedef struct {
    int rank;           /* Current rank */
    int num_procs;      /* total # processors */
    int proc_load;      /* Size of chunks given to each processor */
    int rows;           /* Rows / Cols in A and B */
    int cols;
    int *a_stripe;      /* Partition of A */
    int *b_stripe;      /* Partition of B */
    int *c_stripe;
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
Distriute data to all other processors
*/
void distribute_inital_data(mystery_box_t *my_box, char *a_filename, char *b_filename, int rows, int cols, int num_procs) {
    generate_matrix(rows, cols, a_filename);
    generate_matrix(rows, cols, b_filename);

    int *A = read_matrix(&rows, &cols, a_filename);
    int *B = read_matrix(&rows, &cols, a_filename);
    int proc_load = rows / num_procs; /* # of rows to give to each processor */

    for(int i = 0; i < num_procs; i++) {
        int start = proc_load * i;
        if(i == num_procs-1) {
            if(rows % num_procs != 0) {
                proc_load = rows - start; /* Assign extra rows to last processor */
            }
        }
        int size = proc_load * cols;

        int *little_a = calloc(proc_load * cols, sizeof(int));
        int *little_b = calloc(proc_load * cols, sizeof(int));
        for(int c = 0; c < proc_load*cols; c++) {
            *(little_a + c) = *(A + start*cols + c);
            *(little_b + c) = *(B + start*cols + c);
        }

        // if(DEBUG) {mat_print("a", little_a, proc_load, cols);}
        // if(DEBUG) {mat_print("b", little_b, proc_load, cols);}
        if(i==0) {
            printf("setting 0's data\n");
            // my_box->a_stripe = malloc(sizeof(little_a));
            // my_box->b_stripe = malloc(sizeof(little_b));
            my_box->a_stripe = little_a;
            my_box->b_stripe = little_b;
            my_box->proc_load = proc_load;
            continue;
        }
        if(DEBUG) {printf("%d: proc_load is %d, cols is %d\n", i, proc_load, cols);}
        if(DEBUG) {printf("Sending Data to %d\n", i);}
        MPI_Send(&proc_load, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        if(DEBUG) {printf("Sent proc load to %d\n", i);}
        MPI_Send(little_a, size, MPI_INT, i, 2, MPI_COMM_WORLD);
        if(DEBUG) {printf("Sent a data to %d\n", i);}
        MPI_Send(little_b, size, MPI_INT, i, 3, MPI_COMM_WORLD);
        if(DEBUG) {printf("Sent b data to %d\n", i);}
    }
    // free(A);
    // free(B);
}

/**
Receive data for matrices
*/
void receive_inital_data(mystery_box_t *my_box) {
    MPI_Status status;
    int rows = my_box->rows;
    int cols = my_box->cols;
    int size;
    int rank = my_box->rank;

    MPI_Recv(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    my_box->proc_load = size;
    int *recv_a = malloc(size * cols * sizeof(int));
    int *recv_b = malloc(size * cols * sizeof(int));
    if(DEBUG) {printf("%d: Receiving data from 0\n", my_box->rank);}
    MPI_Recv(recv_a, size * cols, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
    MPI_Recv(recv_b, size * cols, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);

    my_box->a_stripe = recv_a;
    my_box->b_stripe = recv_b;
}

void mat_add (mystery_box_t *my_box) {
    int proc_load = my_box->proc_load;
    int rows = my_box->rows;
    int cols = my_box->cols;
    int size = proc_load * cols;
    int rank = my_box->rank;
    int *c = calloc(size, sizeof(int));
    for(int i = 0; i < size; i++) {
        *(c + i) = *(my_box->a_stripe + i) + *(my_box->b_stripe + i);
    }
    // if(DEBUG) {mat_print("c in progress", c, proc_load, cols);}
    //if(DEBUG) {printf("%d: Sending size of computed data\n", rank);}
    if(rank > 0) {
        if(DEBUG) {printf("%d sending %d\n", rank, size);}
        MPI_Send(&proc_load, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
        if(DEBUG) {printf("%d: Sending computed data, proc_load is %d, size is %d, and cols is %d\n", rank, proc_load, size, cols);}
        MPI_Send(c, size, MPI_INT, 0, 5, MPI_COMM_WORLD);
        if(DEBUG) {printf("%d sent data\n", rank);}
    } else {
        my_box->c_stripe = c;
    }
}

void write_data_to_disk(mystery_box_t *my_box) {
    MPI_Status status;
    int rows = my_box->rows;
    int cols = my_box->cols;
    int num_procs = my_box->num_procs;
    int load = my_box->proc_load;
    FILE *fp;
    if((fp = fopen("c.txt", "w")) == NULL) {
        fprintf(stderr, "Can't open 'c.txt' for writing\n");
        exit(1);
    }
    fprintf(fp, "%d %d\n", rows, cols);
    for(int i = 0; i < load; i++) {
        for(int j = 0; j < cols; j++) {
            fprintf(fp, " %3d", MAT_ELT(my_box->c_stripe, cols, i, j));
        }
        fprintf(fp, "\n");
    }
    for(int i = 1; i < num_procs; i++) {
        int other_size;
        if(DEBUG) {printf("Receiving c matrix data from %d\n", i);}
        MPI_Recv(&other_size, 1, MPI_INT, i, 4, MPI_COMM_WORLD, &status);
        int *c = malloc(cols * other_size * sizeof(int));
        MPI_Recv(c, other_size*cols, MPI_INT, i, 5, MPI_COMM_WORLD, &status);
        int val;
        if(DEBUG) {printf("Writing %d's data to disk\n", i);}
        for(int j = 0; j < other_size; j++) {
            for(int k = 0; k < cols; k++) {
                val = *(c + (j*cols) + k);
                fprintf(fp, " %3d", val);
            }
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
    //mat_print("Final C", read_matrix(&rows, &cols, "c.txt"), rows, cols);
    //free(c);
}

int main(int argc, char **argv) {
    int rows = 60;
    int cols = 60;

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

    /* Set up mystery box */
    mystery_box_t *my_box = malloc(sizeof(mystery_box_t));
    my_box->rows = rows;
    my_box->cols = cols;
    my_box->rank = rank;
    my_box->num_procs = num_procs;

    if(rank == 0) {
        distribute_inital_data(my_box, "a.txt", "b.txt", rows, cols, num_procs);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank != 0){
        receive_inital_data(my_box);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    mat_add(my_box);

    //MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        write_data_to_disk(my_box);
        if(DEBUG) {printf("Finished writing to disk\n");}
    }
    //free(my_box);
    MPI_Finalize();
    if(DEBUG) {printf("Finalized, exiting...\n");}
    return 0;
}
