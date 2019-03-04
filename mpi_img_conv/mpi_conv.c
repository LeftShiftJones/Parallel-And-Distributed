#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include "lodepng.h"

#define true 1
#define false 0
#define RED_OFFSET 0
#define KERNEL_DIM 3
#define BLUE_OFFSET 2
#define GREEN_OFFSET 1
#define ALPHA_OFFSET 3
#define BYTES_PER_PIXEL 4
#define DEFAULT_KERNEL_NAME "identity"
#define ONE_BILLION (double)1000000000.0
#define CLAMP(val, min, max) (val < min ? min : val > max ? max : val)
#define IMG_BYTE(columns, r, c, b) ((columns * BYTES_PER_PIXEL * r) + (BYTES_PER_PIXEL * c) + b)

typedef int bool;
typedef unsigned char pixel_t;
typedef int kernel_t[KERNEL_DIM][KERNEL_DIM];

typedef struct {
    pixel_t *pixels;
    unsigned int rows;
    unsigned int columns;
} image_t;


typedef struct {
    image_t *input;
    image_t *output;
    kernel_t *kernel;
    int thread;
    int num_threads;
} mystery_box_t;


typedef struct {
    char *name;      /* Kernel name */
    kernel_t kernel; /* Kernel itself */
} catalog_entry_t;

catalog_entry_t kernel_catalog[] =
    {
        {DEFAULT_KERNEL_NAME,
         {{0, 0, 0},
          {0, 1, 0},
          {0, 0, 0}}},
        {"edge-detect",
         {{-1, -1, -1},
          {-1, +8, -1},
          {-1, -1, -1}}},
        {"sharpen",
         {{+0, -1, +0},
          {-1, +5, -1},
          {+0, -1, +0}}},
        {"emboss",
         {{-2, -1, +0},
          {-1, +1, +1},
          {+0, -2, +2}}},
        {"gaussian-blur",
         {{1, 2, 1},
          {2, 4, 2},
          {1, 2, 1}}},
        {NULL, {}} /* Must be last! */
};


/**
 * Load image
 */
void load_and_decode(image_t *image, const char *file_name) {
    unsigned int error = lodepng_decode32_file(&image->pixels, &image->columns, &image->rows, file_name);
    if (error) {
        fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
    }
    printf("Loaded %s (%dx%d)\n", file_name, image->columns, image->rows);
}

/**
 * Encode png image into file
 */
void encode_and_store(image_t *image, const char *file_name) {
    unsigned int error = lodepng_encode32_file(file_name, image->pixels, image->columns, image->rows);
    if (error) {
        fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
    }
    //output image info
    printf("Loaded %s (%dx%d)\n", file_name, image->columns, image->rows);
}

void init_image(image_t *image, int rows, int columns) {
    image->rows = rows;
    image->columns = columns;
    image->pixels = (pixel_t *)malloc(image->columns * image->rows * BYTES_PER_PIXEL);
}

/* Free a previously initialized image.
 */
void free_image(image_t *image) {
    free(image->pixels);
}

void copy(image_t *output, image_t *input) {
    int rows = input->rows;
    int columns = input->columns;
    init_image(output, rows, columns);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < columns; c++) {
            for (int b = 0; b < BYTES_PER_PIXEL; b++) {
                output->pixels[IMG_BYTE(columns, r, c, b)] = input->pixels[IMG_BYTE(columns, r, c, b)];
            }
        }
    }
}

int normalize_kernel(kernel_t kernel) {
    int norm = 0;
    for (int r = 0; r < KERNEL_DIM; r++) {
        for (int c = 0; c < KERNEL_DIM; c++) {
            norm += kernel[r][c];
        }
    }
    if (!norm)
        norm = 1;

    return norm;
}


catalog_entry_t* find_entry_by_name(char *name) {
    for (catalog_entry_t *cp = kernel_catalog; cp->name; cp++) {
        if (strcmp(cp->name, name) == 0) {
            return (cp);
        }
    }
    return (catalog_entry_t *)NULL;
}


void distribute_data(image_t *input, char *input_file_name, image_t *output, char *output_file_name) {
    load_and_decode(&input, input_file_name);
    init_image(&output, input.rows, input.columns);
}


void usage(char *prog_name, char *msge) {
    if (msge && strlen(msge)) {
        fprintf(stderr, "\n%s\n\n", msge);
    }

    fprintf(stderr, "usage: %s [flags]\n", prog_name);
    fprintf(stderr, "  -h                   print help\n");
    fprintf(stderr, "  -i <input file>      set input file\n");
    fprintf(stderr, "  -o <output file>     set output file\n");
    fprintf(stderr, "  -k <kernel>          kernel from:\n");
    fprintf(stderr, "  -n <num threads>     # threads to use\n");
    fprintf(stderr, "  -s                   run threads in sequence (1, 2...p)\n");

    for (int i = 0; kernel_catalog[i].name; i++) {
        char *name = kernel_catalog[i].name;
        fprintf(stderr, "       %s%s\n", name,
                strcmp(name, DEFAULT_KERNEL_NAME) ? "" : " (default)");
    }
    exit(1);
}


int main(int argc, char **argv, char **envp) {
    int num_procs;
    int rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    return 0;
}
