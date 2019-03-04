#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "matrix_generator.h"

//#define MAT_ELT(matrix, cols, row, col) *(matrix + (row * cols) + col)

/**
 * Function that takes in
 * three matrices and calculates
 * the addition of the first two.
 */
void mat_add(int *a, int *b, int *c, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            MAT_ELT(c, cols, i, j) = MAT_ELT(a, cols, i, j) + MAT_ELT(b, cols, i, j);
        }
    }
}

/**
 * Method to print out matrices
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
 * Main method of the program
 */
int main(int argc, char **argv) {
    /*
    int A[2][3] = {
        {1, 2, 3},
        {4, 5, 6}
    };

    int B[2][3] = {
        {6, 5, 4},
        {3, 2, 1}
    };

    int C[2][3] = {
        {0, 0, 0},
        {0, 0, 0}
    };
    mat_print("A", (int *)A, 2, 3);
    mat_print("B", (int *)B, 2, 3);

    mat_print("C Before", (int *)C, 2, 3);
    mat_add((int *)A, (int *)B, (int *)C, 2, 3);
    mat_print("C After", (int *)C, 2, 3);
    write_matrix((int *)C, 30, 30)
    */
    int rowcol = 512;
    // generate_matrix(rowcol, rowcol, "a_test.txt");
    // generate_matrix(rowcol, rowcol, "b_test.txt");
    int *a = read_matrix(&rowcol, &rowcol, "a.txt");
    int *b = read_matrix(&rowcol, &rowcol, "b.txt");
    int *c = malloc(rowcol * rowcol * sizeof(int));
    mat_add(a, b, c, rowcol, rowcol);
    write_matrix(c, rowcol, rowcol, "c_test.txt");

    return 0;
}
