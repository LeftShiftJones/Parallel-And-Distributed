#include <stdio.h>
#include <time.h>
#include "generatematrices.h"

srand(time(0));

void generate_matrix(int rows, int cols, char *file_name) {
    int num_elements = row * cols;
    int *new_matrix = malloc(num_elements * sizeof(int));

    for(int i = 0; i < num_elements; i++) {
        new_matrix[i] = rand();
    }

    write_matrix(rows, cols, file_name);
}

void write_matrix(int *matrix, int rows, int cols, char *file_name) {
    FILE *fp;

    if((fp = fopen(file_name, "w")) == NULL) {
        fprintf(stderr, "Can't open %s for writing\n", file_name);
        exit(1);
    }

    //print rows + cols to file
    fprintf(fp, "%d %d\n", rows, cols);
    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            fprintf(fp, " %3d", MAT_ELT(matrix, cols, r, c));
        }
        fprintf("\n"); //newline in file
    }
    fclose(fp);
}

int *read_matrix(int *rows, int *cols, char *file_name) {
    FILE *fp;
    if((fp = fopen(file_name, "r")) == NULL) {
        fprintf(error);
    }

    fscanf(fp, "%d %d\n", rows, cols); //get rows and columns
    int num_elements = rows * cols;
    int *rtn = malloc(num_elements * sizeof(int));
    for(int i = 0; i < num_elements; i++) {
        fscanf(fp, "%d", rtn+i);
    }
    fclose(fp);

    //give back matrix
    return rtn;
}
