#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "generatematrices.h"

/**
Matrix generation file,
"inspired" by Dr. Nurkkala's in-class demonstration
*/

void generate_matrix(int rows, int cols, char *file_name) {
    //srand(time(0));
    int num_elements = rows*cols;
    int *new_matrix = malloc(num_elements * sizeof(int));

    for(int i = 0; i < num_elements; i++) {
        new_matrix[i] = random() / (RAND_MAX / 100);
    }

    write_matrix(new_matrix, rows, cols, file_name);
}

void write_matrix(int *matrix, int rows, int cols, char *file_name) {
    FILE *fp;

    if((fp = fopen(file_name, "w")) == NULL) {
        fprintf(stderr, "Can't open %s for writing\n", file_name);
        exit(1);
    }

    fprintf(fp, "%d %d\n", rows, cols);
    for(int r = 0; r < rows; r++) {
        for(int c = 0; c < cols; c++) {
            fprintf(fp, " %3d", MAT_ELT(matrix, cols, r, c));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int *read_matrix(int *rows, int *cols, char *file_name) {
    FILE *fp;
    if((fp = fopen(file_name, "r")) == NULL) {
        fprintf(stderr, "Problem parsing file.\n");
        exit(1);
    }

    fscanf(fp, "%d %d\n", rows, cols);
    int num_elements = *rows * *cols;
    int *rtn = malloc(num_elements * sizeof(int));
    for(int i = 0; i < num_elements; i++) {
        fscanf(fp, "%d", rtn+i);
    }
    fclose(fp);

    return rtn;
}
