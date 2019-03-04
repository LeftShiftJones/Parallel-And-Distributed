#include <stdio.h>
#include <stdlib.h>

/**** List ADT ****************/

typedef struct
{
    int *values;  /* Values stored in list */
    int max_size; /* Maximum size allocated */
    int cur_size; /* Size currently in use */
} list_t;

/* Dump list, including sizes */
void list_dump(list_t *list) {
    printf("%2d/%2d", list->cur_size, list->max_size);
    for (int i = 0; i < list->cur_size; i++) {
        printf(" %d", list->values[i]);
    }
    printf("\n");
}

/* Allocate list that can store up to 'max_size' elements */
list_t *
list_alloc(int max_size) {
    list_t *list = (list_t *)malloc(sizeof(list_t));
    list->values = (int *)malloc(max_size * sizeof(int));
    list->max_size = max_size;
    list->cur_size = 0;
    return list;
}

/* Free a list; call this to avoid leaking memory! */
void list_free(list_t *list) {
    free(list->values);
    free(list);
}

/* Add a value to the end of the list */
void list_add(list_t *list, int value) {
    if (list->cur_size >= list->max_size) {
        printf("List full");
        list_dump(list);
        exit(1);
    }
    list->values[list->cur_size++] = value;
}

/* Return the current size of the list */
int list_size(list_t *list) {
    return list->cur_size;
}

/* Validate index */
void _list_check_index(list_t *list, int index) {
    if (index < 0 || index > list->cur_size - 1) {
        printf("Invalid index %d\n", index);
        list_dump(list);
        exit(1);
    }
}

/* Get the value at given index */
int list_get(list_t *list, int index) {
    _list_check_index(list, index);
    return list->values[index];
}

/* Remove the value at the given index */
void list_remove_at(list_t *list, int index) {
    _list_check_index(list, index);
    for (int i = index; i < list->cur_size - 1; i++) {
        list->values[i] = list->values[i + 1];
    }
    list->cur_size--;
}

/* Retrieve a copy of the values as a simple array of integers. The returned
   array is allocated dynamically; the caller must free the space when no
   longer needed.
 */
int *list_as_array(list_t *list) {
    int *rtn = (int *)malloc(list->max_size * sizeof(int));
    for (int i = 0; i < list->max_size; i++) {
        rtn[i] = list_get(list, i);
    }
    return rtn;
}

/**** Permutation ****************/

/* Permutation algorithms based on code found at:
   http://www.mathblog.dk/project-euler-24-millionth-lexicographic-permutation/
   which references:
   http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
*/

/* Calculate n! iteratively */
long factorial(int n) {
    if (n < 1) {
        return 0;
    }

    long rtn = 1;
    for (int i = 1; i <= n; i++) {
        rtn *= i;
    }
    return rtn;
}

/* Return the kth lexographically ordered permuation of an array of k integers
   in the range [0 .. size - 1]. The integers are allocated dynamically and
   should be free'd by the caller when no longer needed.
*/
int *kth_perm(int k, int size) {
    long remain = k - 1;

    list_t *numbers = list_alloc(size);
    for (int i = 0; i < size; i++) {
        list_add(numbers, i);
    }

    list_t *perm = list_alloc(size);

#if DEBUG
    printf("k=%d, size=%d, remain=%ld\n", k, size, remain);
    printf("  perm");
    list_dump(perm);
    printf("  nums");
    list_dump(numbers);
#endif

    for (int i = 1; i < size; i++) {
        long f = factorial(size - i);
        long j = remain / f;
        remain = remain % f;
#if DEBUG
        printf("i=%d, f=%ld j=%ld, remain=%ld\n", i, f, j, remain);
#endif

        list_add(perm, list_get(numbers, j));
        list_remove_at(numbers, j);

#if DEBUG
        printf("  perm");
        list_dump(perm);
        printf("  nums");
        list_dump(numbers);
#endif

        if (remain == 0) {
            break;
        }
    }

    /* Append remaining digits */
    for (int i = 0; i < list_size(numbers); i++) {
        list_add(perm, list_get(numbers, i));
    }

    int *rtn = list_as_array(perm);
    list_free(perm);

    return rtn;
}

/* Swap v[i] and v[j] */
void swap(int *v, int i, int j) {
    int t = v[i];
    v[i] = v[j];
    v[j] = t;
}

/* Print a permutation array */
void print_perm(int *perm, int size) {
    for (int k = 0; k < size; k++) {
        printf("%4d", perm[k]);
    }
    printf("\n");
}

/* Given an array of size elements at perm, update the array in place to
   contain the lexographically next permutation. It is originally due to
   Dijkstra. The present version is discussed at:
   http://www.cut-the-knot.org/do_you_know/AllPerm.shtml
 */
void next_perm(int *perm, int size) {
    int i = size - 1;
    while (perm[i - 1] >= perm[i]) {
        i = i - 1;
    }

    int j = size;
    while (perm[j - 1] <= perm[i - 1]) {
        j = j - 1;
    }

    swap(perm, i - 1, j - 1);

    i++;
    j = size;
    while (i < j) {
        swap(perm, i - 1, j - 1);
        i++;
        j--;
    }
}

/* Calculate the kth permuation of an array of size integers. This function
   computes the same thing as kth_perm(), but uses a brute-force algorithm. It
   serves to cross validate the "kth perm" calculation and test the next_perm()
   function.
*/
void kth_perm_brute(int k, int size) {
    int *perm = malloc(size * sizeof(int));
    for (int i = 0; i < size; i++) {
        perm[i] = i;
    }

    int count = 1;
    while (count < k) {
        next_perm(perm, size);
        count++;
    }

    print_perm(perm, size);

    free(perm);
}

/**** Tests ****************/

void test_kth_perm(void) {
    for (int size = 1; size <= 5; size++) {
        printf("Size = %d\n", size);
        for (int k = 1; k <= factorial(size); k++) {
            int *perm = kth_perm(k, size);
            printf("K = %4d    ", k);
            print_perm(perm, size);
            free(perm);
        }
    }
}

/**** Driver ****************/

int main(int argc, char **argv) {
    const int k = 1000000;
    const int size = 10;

    test_kth_perm();

    printf("\nFACTORIALS\n");
    for (int i = 0; i < 24; i++) {
        long f = factorial(i);
        printf("%2d! = %20ld ~ %12.4g\n", i, f, (double)f);
    }

    printf("\nBRUTE\n");
    kth_perm_brute(k, size);

    printf("\nCOMBINATORIC\n");
    int *perm = kth_perm(k, size);
    print_perm(perm, size);

    printf("\nNEXT FEW\n");
    for (int i = 0; i < 20; i++) {
        next_perm(perm, size);
        print_perm(perm, size);
    }

    printf("\nTSP\n");
    const long cities = 20; /* Max of 20; else factorial overflows */
    const long cores = 1024;
    const long tours = factorial(cities);
    const long tours_per_core = tours / cores;
    printf("%ld cities\n", cities);
    printf("%ld cores\n", cores);
    printf("%20ld ~ %.4g tours\n", tours, (double)tours);
    printf("%20ld ~ %.4g tours/core\n", tours_per_core, (double)tours_per_core);

    for (int c = 0; c < cores; c += cores / 8) {
        long start_tour = tours_per_core * c;
        long end_tour = start_tour + tours_per_core - 1;
        printf("%4d: %19ld - %19ld\n", c, start_tour, end_tour);
    }
}
