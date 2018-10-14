#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#include "generatematrices.h"

#define MAT_ELT(mat, cols, i, j) *(mat + (i * cols) + j)
typedef int bool;
#define true 1
#define false 0
#define MASTER_CORE 0
#define DEFAULT_TAG 1
#define ONE_BILLION (double)1000000000.0

/* object to store important information */
typedef struct {
    // int a_load;
    // int b_load;
    int *a_stripe;      /* Stripe of A */
    int *b_stripe;      /* Stripe of B */
    //int *c_stripe;      /* Stripe of C */
} mystery_box_t;

/**
 * Method to print out matrices
 * !Provided by Dr. Nurkkala!
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
 * Custom method for transposing a matrix along the diagonal
 */
int *transpose_matrix(int *matrix, int rows, int cols) {
    int length = rows * cols;
    int *rtn = calloc(length, sizeof(matrix));
    for(int i = 0; i < cols; i++) {
        for(int j = 0; j < rows; j++) {
            MAT_ELT(rtn, rows, i, j) = MAT_ELT(matrix, cols, j, i);
        }
    }
    return rtn;
}

/**
    Old method for matrix multiplication
*/
void seq_mat_mult(int *c, int *a, int *b, int m, int n, int p) {
    for (int i = 0;  i < m;  i++) {
        for (int j = 0;  j < p;  j++) {
            for (int k = 0;  k < n;  k++) {
                MAT_ELT(c, p, i, j) += MAT_ELT(a, n, i, k) * MAT_ELT(b, n, j, k);
            }
        }
    }
}

/**
 * Main matrix multiplication method
 */
void mat_mult(mystery_box_t *box, int this_rank, int procs, int a_load, int b_load, int m, int n, int p) {
    // int a_load = box->a_load;
    // int b_load = box->b_load;
    int *a = box->a_stripe;
    int *b = box->b_stripe;
    int *c = calloc(a_load*p, sizeof(int));//box->c_stripe;
    //mat_print("a", a, a_load, n);
    //mat_print("b", b, b_load, n);
    MPI_Status status;
    //set next and previous processors
    int next_proc, prev_proc;
    next_proc = (this_rank + 1) % procs;
    prev_proc = (!this_rank) ? procs-1 : this_rank-1;
    int *not_mine = malloc(b_load*n * sizeof(int));
    //printf("Starting loop...\n");
    //////////BEGINNING OF LOOP//////////
    for(int step = 0; step < procs; step++) {

    for (int i = 0;  i < a_load;  i++) {
        for (int j = 0;  j < b_load;  j++) {
            for (int k = 0;  k < n;  k++) {
                int loc = (j + (((this_rank + step) % procs) * b_load)); // <-- This beauty is courtesy of Benj R.
                //printf("%d: %d = %d x %d[%d][%d]\n", this_rank, MAT_ELT(c, p, i, loc), MAT_ELT(a, n, i, k), MAT_ELT(b, n, j, k), j, k);
                MAT_ELT(c, p, i, loc) += MAT_ELT(a, n, i, k) * MAT_ELT(b, n, j, k);
            }
        }
    }
    //printf("%d: performing send %d\n", this_rank, (step+1));
    //send/receive c values
    if(procs > 1) {
        MPI_Send(b,
            b_load*n, MPI_INT,
            next_proc, DEFAULT_TAG,
            MPI_COMM_WORLD);
        MPI_Recv(not_mine,
            b_load*n, MPI_INT,
            prev_proc, DEFAULT_TAG,
            MPI_COMM_WORLD, &status);
        //printf("receiveing b\n");
        b = not_mine;
    }

    }
    //////////END OF LOOP//////////

    MPI_Barrier(MPI_COMM_WORLD);
    //send all data
    if(this_rank != MASTER_CORE) {
        MPI_Send(c,
            a_load*p, MPI_INT,
            MASTER_CORE, DEFAULT_TAG,
            MPI_COMM_WORLD);
            //printf("Sent the data\n");
    }
    MPI_Barrier;
    //as rank 0, write data to disk;
    if(this_rank == 0) {
        //output filename
        char *file_name = "c_distributed.txt";
        FILE *fp;

        //test opening file
        if((fp = fopen(file_name, "w")) == NULL) {
            fprintf(stderr, "Can't open %s for writing\n", file_name);
            exit(1);
        }
        //print rows and cols to file
        //print processor 0's c chunk
        fprintf(fp, "%d %d\n", m, p);
        for(int i = 0; i < a_load; i++) {
            for(int j = 0; j < p; j++) {
                fprintf(fp, " %3d", MAT_ELT(c, p, i, j));
            }
            fprintf(fp, "\n");
        }
        printf("About to bring in values\n");
        //start getting values from other processors and
        int *other_c_rows = malloc(a_load*p*sizeof(int));
        for(int i = 1; i < procs; i++) {
            MPI_Recv(other_c_rows,
                a_load*p, MPI_INT,
                i, DEFAULT_TAG,
                MPI_COMM_WORLD, &status);
            //mat_print("c_received", other_c_rows, a_load, p);
            for(int j = 0; j < a_load; j++) {
                for(int k = 0; k < p; k++) {
                    fprintf(fp, " %3d", MAT_ELT(other_c_rows, p, j, k));
                }
                fprintf(fp, "\n");
            }
        }
        free(other_c_rows);
        fclose(fp);
        //mat_print("Multiplied", read_matrix(&m, &p, file_name), m, p);
    }
    MPI_Barrier;
}

/**
 * Prints out program usage information
 */
void usage(char *prog_name, char *msg) {
    if(msg && strlen(msg)) {
        fprintf(stderr, "\n%s\n\n", msg);
    }
    fprintf(stderr, "usage: %s [flags]\n", prog_name);
    fprintf(stderr, "   -h                  print help\n");
    fprintf(stderr, "   -m  <value>         m value for matrix generation\n");
    fprintf(stderr, "   -n  <value>         n value for matrix generation\n");
    fprintf(stderr, "   -p  <value>         p value for matrix generation\n");
    fprintf(stderr, "   -a  <a_matrix>      name of file for a matrix\n");
    fprintf(stderr, "   -b  <b_matrix>      name of file for b matrix\n");
    fprintf(stderr, "   -o  <o_filename>    name of file for output matrix\n");
    exit(1);
}

/**
 * Method for getting the current time
 */
double now(void) {
    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

int main(int argc, char **argv) {
    char *prog_name = argv[0];
    int ch;
    int m = 64;
    int n = 64;
    int p = 64;
    char *a_filename;
    char *b_filename;
    char *c_filename;

    while((ch = getopt(argc, argv, "hm:n:p:a:b:o:")) != -1) {
        switch(ch) {
            case 'm':
                m = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'p':
                p = atoi(optarg);
                break;
            case 'a':
                a_filename = optarg;
                break;
            case 'b':
                b_filename = optarg;
                break;
            case 'o':
                c_filename = optarg;
                break;
            case 'h':
            default:
                usage(prog_name, "");
        }
    }
    if(!a_filename || !b_filename || !c_filename) usage(prog_name, "No file(s) specified");
    if(m < 1 || n < 1 || p < 1) usage(prog_name, "Invalid m, p, or n values");
    if(p%2==1) usage(prog_name, "P cannot be odd");

    //MPI Stuff
    int num_procs;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;
    if(num_procs > m || num_procs > p) usage(prog_name,
        "Too many processors for the number of rows/columns specified.\n");
    mystery_box_t *my_box = malloc(sizeof(mystery_box_t));

    //stuff to give to each process
    int a_load = (m / num_procs);
    int b_load = (p / num_procs);
    int a_start;
    int b_start;

    //only processor 0 generates values, distributes
    if(rank == MASTER_CORE) {
        generate_matrix(m, n, a_filename);
        generate_matrix(n, p, b_filename);

        //import matrices
        int *matrix_a = read_matrix(&m, &n, a_filename);
        int *matrix_b = transpose_matrix(read_matrix(&n, &p, b_filename), n, p);
        int *a = calloc(a_load*n, sizeof(int));
        int *b = calloc(b_load*n, sizeof(int));
        int *c = calloc(a_load*p, sizeof(int));

        for(int i = 1; i < num_procs; i++) {
            a_start = a_load * i;
            b_start = b_load * i;
            //fill values for a and b stripes
            for(int j = 0; j < a_load*n; j++) {
                *(a + j) = *(matrix_a + (j + (n*a_start)));
            }
            for(int j = 0; j < b_load*n; j++) {
                *(b + j) = *(matrix_b + (j + (n*b_start)));
            }
            MPI_Send(a, a_load*n, MPI_INT, i, DEFAULT_TAG, MPI_COMM_WORLD);
            MPI_Send(b, b_load*n, MPI_INT, i, DEFAULT_TAG, MPI_COMM_WORLD);
        }

        //set up cpu 0 values
        a_start = 0;
        b_start = 0;
        for(int j = 0; j < a_load*n; j++) {
            *(a + j) = *(matrix_a + (j + (n*a_start)));
        }
        for(int j = 0; j < b_load*n; j++) {
            *(b + j) = *(matrix_b + (j + (n*b_start)));
        }
        my_box->a_stripe = a;
        my_box->b_stripe = b;
    }
    //for all other processors:
    if(rank != 0) {
        int *get_a = calloc(a_load*n, sizeof(int));
        int *get_b = calloc(b_load*n, sizeof(int));
        MPI_Recv(get_a, a_load*n, MPI_INT, 0, DEFAULT_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(get_b, b_load*n, MPI_INT, 0, DEFAULT_TAG, MPI_COMM_WORLD, &status);
        my_box->a_stripe = get_a;
        my_box->b_stripe = get_b;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = now();
    //perform matrix multiplication
    mat_mult(my_box, rank, num_procs, a_load, b_load, m, n, p);
    if(!rank) {
        printf("With %d cores, calculating an %dx%d matrix took %5.3f seconds\n", num_procs, m, p, now()-start_time);
    }
    MPI_Finalize();
}
