#include <stdio.h>
#include "generatematrices.h"

#define MAT_ELT(mat, cols, i, j) *(mat + (i * cols) + j)

void
mat_print(char *msge, int *a, int m, int n)
{
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

void
mat_mult(int *c, int *a, int *b, int m, int n, int p)
{
  for (int i = 0;  i < m;  i++) {
    for (int j = 0;  j < p;  j++) {
      for (int k = 0;  k < n;  k++) {
        MAT_ELT(c, p, i, j) += MAT_ELT(a, n, i, k) * MAT_ELT(b, p, k, j);
      }
    }
  }
}

int
main(int argc, char **argv)
{
  int A[3][4] = {
    { 1, 3, 6, 7 },
    { 4, 6, 3, 9 },
    { 5, 5, 0, 2 }
  };

  int B[4][2] = {
    { 4, 2 },
    { 6, 3 },
    { 6, 9 },
    { 0, 9 }
  };

  int C[3][2] = {
    { 0, 0 },
    { 0, 0 },
    { 0, 0 }
  };

  mat_mult((int *)C, (int *)A, (int *)B, 3, 4, 2);
  mat_print("A", (int *)A, 3, 4);
  mat_print("B", (int *)B, 4, 2);
  mat_print("C", (int *)C, 3, 2);

  generate_matrix(3, 5, "matrix_a.mat");
  int *test = read_matrix(3, 5, "matrix_a.mat")
  mat_print("TEST", test, 3, 5);
}
