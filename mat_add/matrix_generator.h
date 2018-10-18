#define MAT_ELT(mat, cols, i, j) *(mat + (i * cols) + j)

extern void write_matrix(int *matrix, int rows, int cols, char *file_name);
extern void generate_matrix(int rows, int cols, char *file_name);
extern int *read_matrix(int *rows, int *cols, char *file_name);
