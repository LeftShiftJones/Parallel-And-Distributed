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

/* object to store important information */
typedef struct {
    int rank;           /* Current process rank */
    int num_procs;      /* Total # of processes */
    int a_rows;         /* Rows in A */
    int a_cols;         /* Cols in A */
    int b_rows;         /* Rows in B */
    int b_cols;         /* Cols in B */
    int c_rows;         /* Rows in C */
    int c_cols;         /* Cols in C */

    int a_load;         /* Number of rows of A per processor */
    int b_load;         /* Number of rows of B per processor */
    int c_load;         /* Number of rows of C per processor */

    int *a_stripe;      /* Stripe of A */
    int *b_stripe;      /* Stripe of B */
    int *c_stripe;      /* Stripe of C */
} mystery_box_t;


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
*
void amat_mult(int *c, int *a, int *b, int m, int n, int p) {
    for (int i = 0;  i < m;  i++) {
        for (int j = 0;  j < p;  j++) {
            for (int k = 0;  k < n;  k++) {
                MAT_ELT(c, p, i, j) += MAT_ELT(a, n, i, k) * MAT_ELT(b, p, k, j);
            }
        }
    }
} */

void mat_mult(mystery_box_t *box, int rank, int procs) {
    int m = box->a_rows;
    int n = box->a_cols;
    int p = box->b_rows;
    int a_load = box->a_load;
    int b_load = box->b_load;
    int c_load = box->c_load;
    int *a = box->a_stripe;
    int *b = box->b_stripe;
    int *c = box->c_stripe;
    printf("Processor %d of %d checking in\n", rank, procs);
    //set next and previous processors
    int next_proc, prev_proc;
    next_proc = (rank + 1) % procs;
    prev_proc = (!rank) ? procs-1 : rank-1;


    for (int i = 0;  i < m && i < a_load;  i++) {
        for (int j = 0;  j < p && j < b_load;  j++) {
            for (int k = 0;  k < n;  k++) {
                MAT_ELT(c, p, i, j) += MAT_ELT(a, n, i, k) * MAT_ELT(b, p, j, k);
            }
        }
    }
    printf("We somehow made it here\n");
    MPI_Barrier(MPI_COMM_WORLD);
    printf("getting out, probably about to error");
}

void usage(char *prog_name, char *msg) {
    if(msg && strlen(msg)) {
        fprintf(stderr, "\n%s\n\n", msg);
    }

    fprintf(stderr, "usage: %s [flags]\n", prog_name);
    fprintf(stderr, "   -h                  print help\n");
    fprintf(stderr, "   -g                  generate matrix files (otherwise read from given filenames)\n");
    fprintf(stderr, "   -m  <value>         m value for matrix generation\n");
    fprintf(stderr, "   -n  <value>         n value for matrix generation\n");
    fprintf(stderr, "   -p  <value>         p value for matrix generation\n");
    fprintf(stderr, "   -a  <a_matrix>      name of file for a matrix\n");
    fprintf(stderr, "   -b  <b_matrix>      name of file for b matrix\n");
    fprintf(stderr, "   -o  <o_filename>    name of file for output matrix\n");
    exit(1);
}

int main(int argc, char **argv) {
    char *prog_name = argv[0];
    int ch;
    int m = 3;
    int n = 4;
    int p = 5;
    char *a_filename;
    char *b_filename;
    char *c_filename;
    bool generate_files = false;
    while((ch = getopt(argc, argv, "hgm:n:p:a:b:o:")) != -1) {
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
            case 'g':
                generate_files = true;
                break;
            case 'h':
            default:
                usage(prog_name, "");
        }
    }
    if(!a_filename || !b_filename || !c_filename) {
        usage(prog_name, "No file(s) specified");
    }
    if(m < 1 || n < 1 || p < 1) {
        usage(prog_name, "Invalid m, p, or n values");
    }

    //create a and b matrices if specified
    if(generate_files) {
        generate_matrix(m, n, a_filename);
        generate_matrix(n, p, b_filename);
    }

    //import matrices
    int num_c_elements = m*p;
    int *matrix_a = read_matrix(&m, &n, a_filename);
    int *matrix_b = transpose_matrix(read_matrix(&n, &p, b_filename), n, p);
    int *matrix_c = calloc(num_c_elements, sizeof(int));
    mat_print("C", matrix_c, m, p);

    //MPI Stuff

    int num_procs;
    int rank;

    //stuff to give to each process
    int a_load = (m / num_procs);
    int b_load = (p / num_procs);

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Matrix multiplying on processor %d of %d\n", rank, num_procs);

    mystery_box_t *box = malloc(sizeof(mystery_box_t));
    box->rank = rank;
    box->num_procs = num_procs;
    box->a_rows = m;
    box->a_cols = n;
    box->b_rows = p;
    box->b_cols = n;
    box->c_rows = m;
    box->c_cols = p;
    box->a_load = a_load;
    box->b_load = b_load;
    box->c_load = a_load;
    box->a_stripe = &MAT_ELT(matrix_a, n, (rank*a_load), 0);
    box->b_stripe = &MAT_ELT(matrix_b, n, (rank*b_load), 0);
    box->c_stripe = &MAT_ELT(matrix_c, p, (rank*a_load), 0);
    mat_mult(box, rank, num_procs);
    //mat_mult(rank, num_procs, matrix_c, matrix_a, matrix_b, m, n, p);
    MPI_Finalize();
    //matrix_b = transpose_matrix(matrix_b, )
    if(!rank) {
        write_matrix(matrix_c, m, p, c_filename);
        mat_print("A", matrix_a, m, n);
        mat_print("B", matrix_b, n, p);
        mat_print("C", matrix_c, m, p);
    }
}
