#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#include "generatematrices.h"

#define MAT_ELT(mat, cols, i, j) *(mat + (i * cols) + j)

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

void mat_mult(int *c, int *a, int *b, int m, int n, int p){
    for (int i = 0;  i < m;  i++) {
        for (int j = 0;  j < p;  j++) {
            for (int k = 0;  k < n;  k++) {
                MAT_ELT(c, p, i, j) += MAT_ELT(a, n, i, k) * MAT_ELT(b, p, k, j);
            }
        }
    }
}

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

int main(int argc, char **argv) {
    char *prog_name = argv[0];
    int ch;
    int m = 3;
    int n = 4;
    int p = 5;
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
    if(!a_filename || !b_filename || !c_filename) {
        usage(prog_name, "No file(s) specified");
    }
    if(m < 1 || n < 1 || p < 1) {
        usage(prog_name, "Invalid mpn values");
    }

    //create a and b matrices
    generate_matrix(m, n, a_filename);
    generate_matrix(n, p, b_filename);
    int num_c_elements = m*p;
    int *matrix_a = read_matrix(&m, &n, a_filename);
    int *matrix_b = read_matrix(&n, &p, b_filename);



    int *matrix_c = calloc(num_c_elements, sizeof(int));

    int num_procs;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Matrix multiplying on processor %d of %d\n", rank, num_procs);
    mat_mult(matrix_c, matrix_a, matrix_b, m, n, p);
    MPI_Finalize();

    write_matrix(matrix_c, m, p, c_filename);
    mat_print("A", matrix_a, m, n);
    mat_print("B", matrix_b, n, p);
    mat_print("C", matrix_c, m, p);


    //old version of matrix that didn't work, created elements prior to multiplication
    // int matrix_c[m][p];
    // for(int i=0; i<m; i++) {
    //     for(int j=0; j<p; j++) {
    //         matrix_c[i][j] = 0;
    //     }
    // }
    // mat_mult((int *)matrix_c, matrix_a, matrix_b, m, n, p);
    // write_matrix((int *)matrix_c, m, p, c_filename);
    // mat_print("A", matrix_a, m, n);
    // mat_print("B", matrix_b, n, p);
    // mat_print("C", (int *)matrix_c, m, p);
}
