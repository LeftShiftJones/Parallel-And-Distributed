#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#include "lodepng.h"

#define BYTES_PER_PIXEL 4
#define RED_OFFSET 0
#define GREEN_OFFSET 1
#define BLUE_OFFSET 2
#define ALPHA_OFFSET 3
#define ONE_BILLION (double)1000000000.0

#define IMG_BYTE(columns, r, c, b) ((columns * BYTES_PER_PIXEL * r) + (BYTES_PER_PIXEL * c) + b)
#define CLAMP(val, min, max) (val < min ? min : val > max ? max : val)

typedef unsigned char pixel_t;

typedef int bool;
#define true 1
#define false 0

typedef struct
{
    pixel_t *pixels;
    unsigned int rows;
    unsigned int columns;
} image_t;

#define KERNEL_DIM 3
typedef int kernel_t[KERNEL_DIM][KERNEL_DIM];

typedef struct
{
    image_t *input;
    image_t *output;
    kernel_t *kernel;
    int thread;
    int num_threads;
} mystery_box_t;

/**
 * Load image
 */
void load_and_decode(image_t *image, const char *file_name)
{
    unsigned int error = lodepng_decode32_file(&image->pixels, &image->columns, &image->rows, file_name);
    if (error)
    {
        fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
    }
    printf("Loaded %s (%dx%d)\n", file_name, image->columns, image->rows);
}

/**
 * Encode png image into file
 */
void encode_and_store(image_t *image, const char *file_name)
{
    unsigned int error = lodepng_encode32_file(file_name, image->pixels, image->columns, image->rows);
    if (error)
    {
        fprintf(stderr, "error %u: %s\n", error, lodepng_error_text(error));
    }
    //output image info
    printf("Loaded %s (%dx%d)\n", file_name, image->columns, image->rows);
}

void init_image(image_t *image, int rows, int columns)
{
    image->rows = rows;
    image->columns = columns;
    image->pixels = (pixel_t *)malloc(image->columns * image->rows * BYTES_PER_PIXEL);
}

/* Free a previously initialized image.
 */
void free_image(image_t *image)
{
    free(image->pixels);
}

void copy(image_t *output, image_t *input)
{
    int rows = input->rows;
    int columns = input->columns;
    init_image(output, rows, columns);

    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < columns; c++)
        {
            for (int b = 0; b < BYTES_PER_PIXEL; b++)
            {
                output->pixels[IMG_BYTE(columns, r, c, b)] = input->pixels[IMG_BYTE(columns, r, c, b)];
            }
        }
    }
}

int normalize_kernel(kernel_t kernel)
{
    int norm = 0;
    for (int r = 0; r < KERNEL_DIM; r++)
    {
        for (int c = 0; c < KERNEL_DIM; c++)
        {
            norm += kernel[r][c];
        }
    }
    if (!norm)
        norm = 1;

    return norm;
}

void *convolve(void *thing)
{
    //cast void * to my mystery_box_t object
    mystery_box_t *box = (mystery_box_t *)thing;
    image_t *input = box->input;
    image_t *output = box->output;
    kernel_t *kernel = box->kernel;
    int columns = input->columns;
    int rows = input->rows;
    int half_dim = KERNEL_DIM / 2;
    int kernel_norm = normalize_kernel(*kernel);

    //Go through image by rows, jumping by the number of threads
    for (int r = box->thread; r < rows; r += box->num_threads)
    {
        for (int c = 0; c < columns; c++)
        {
            for (int b = 0; b < BYTES_PER_PIXEL; b++)
            {
                //Don't alter alpha values
                int value = 0;
                if (b == ALPHA_OFFSET)
                {
                    value = input->pixels[IMG_BYTE(columns, r, c, b)];
                }
                else
                {
                    for (int kr = 0; kr < KERNEL_DIM; kr++)
                    {
                        for (int kc = 0; kc < KERNEL_DIM; kc++)
                        {
                            /**
                             * Here is where I handle the edges, which previously were
                             * filled with an empty alpha channel.
                             */
                            //temp value for r & c
                            int substitute_r = r;
                            int substitute_c = c;
                            //alter temp values if at edges of image
                            //only runs at edges, no less efficient than original program
                            if(r==0) {
                                substitute_r++;
                                if(c==0) { substitute_c++; }
                                if(c==columns-1) { substitute_c--; }
                            }
                            if(r==rows-1){
                                substitute_r--;
                                if(c==0) { substitute_c++; }
                                if(c==columns-1) { substitute_c--; }
                            }
                            //alter value with kernel value, adjusted for edges of frame
                            int R = substitute_r + (kr - half_dim);//out of bounds here at r == 0
                            int C = substitute_c + (kc - half_dim);
                            value += (*kernel)[kr][kc] * input->pixels[IMG_BYTE(columns, R, C, b)];
                        }
                    }
                    value /= kernel_norm;
                    value = CLAMP(value, 0, 0xFF);
                }
                output->pixels[IMG_BYTE(columns, r, c, b)] = value;
            }
        }
    }

    return (void *)NULL;
}

#define DEFAULT_KERNEL_NAME "identity"

typedef struct
{
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

catalog_entry_t *
find_entry_by_name(char *name)
{
    for (catalog_entry_t *cp = kernel_catalog; cp->name; cp++)
    {
        if (strcmp(cp->name, name) == 0)
        {
            return (cp);
        }
    }
    return (catalog_entry_t *)NULL;
}

/* Print an optional message, usage information, and exit in error.
 */
void usage(char *prog_name, char *msge)
{
    if (msge && strlen(msge))
    {
        fprintf(stderr, "\n%s\n\n", msge);
    }

    fprintf(stderr, "usage: %s [flags]\n", prog_name);
    fprintf(stderr, "  -h                   print help\n");
    fprintf(stderr, "  -i <input file>      set input file\n");
    fprintf(stderr, "  -o <output file>     set output file\n");
    fprintf(stderr, "  -k <kernel>          kernel from:\n");
    fprintf(stderr, "  -n <num threads>     # threads to use\n");
    fprintf(stderr, "  -s                   run threads in sequence (1, 2...p)\n");

    for (int i = 0; kernel_catalog[i].name; i++)
    {
        char *name = kernel_catalog[i].name;
        fprintf(stderr, "       %s%s\n", name,
                strcmp(name, DEFAULT_KERNEL_NAME) ? "" : " (default)");
    }
    exit(1);
}

/**
 * Method to initalize a new myster box struct
 */
mystery_box_t *create_mystery_box(image_t *input, image_t *output, int current_thread, int num_threads, kernel_t *kernel)
{
    //allocated memory is later freed
    mystery_box_t *new = malloc(sizeof(mystery_box_t));
    new->input = input;
    new->output = output;
    new->kernel = kernel;
    new->thread = current_thread;
    new->num_threads = num_threads;
    return new;
}

//get current time
double now(void) {
    struct timespec current_time;
    clock_gettime(CLOCK_REALTIME, &current_time);
    return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

/**
 * Main method of program
 */
int main(int argc, char **argv)
{
    char *prog_name = argv[0]; /* Convenience */
#define ERR_MSGE_LEN 128
    char err_msge[ERR_MSGE_LEN]; /* Dynamic error messages */

    catalog_entry_t *selected_entry = find_entry_by_name(DEFAULT_KERNEL_NAME);
    char *input_file_name = NULL;
    char *output_file_name = NULL;
    int num_threads_used = 1;
    bool run_sequence = false;

    int ch;
    while ((ch = getopt(argc, argv, "hi:k:o:n:s")) != -1)
    {
        switch (ch)
        {
        case 'i':
            input_file_name = optarg;
            break;
        case 'k':
            selected_entry = find_entry_by_name(optarg);
            if (selected_entry == (catalog_entry_t *)NULL)
            {
                snprintf(err_msge, ERR_MSGE_LEN, "no kernel named '%s'", optarg);
                usage(prog_name, err_msge);
            }
            break;
        case 'o':
            output_file_name = optarg;
            break;
        case 'n':
            num_threads_used = atoi(optarg);
            break;
        case 's':
            run_sequence = true;
            break;
        case 'h':
        default:
            usage(prog_name, "");
        }
    }

    if (!input_file_name)
    {
        usage(prog_name, "No input file specified");
    }
    if (!output_file_name)
    {
        usage(prog_name, "No output file specified");
    }
    if (strcmp(input_file_name, output_file_name) == 0)
    {
        usage(prog_name, "Input and output file can't be the same");
    }
    if (num_threads_used < 1)
    {
        usage(prog_name, "Invalid number of threads");
    }

    image_t input;
    image_t output;
    pthread_t threads[num_threads_used];
    //mystery box for every thread being run
    mystery_box_t *boxes[num_threads_used];

    //bring in input image and initialize output image
    load_and_decode(&input, input_file_name);
    init_image(&output, input.rows, input.columns);

    /**
     * This portion of the code runs image convolution for a number of
     * processors in sequence (convolve on 1, 2, ... n)
     * I created this functionality mostly for testing runtimes on
     * various processor sizes at once
     */
    if(run_sequence)
    {
        for(int c = 1; c <= num_threads_used; c++)
        {
            for (int i = 0; i < c; i++)
            {
                //initalize all mystery boxes
                boxes[i] = create_mystery_box(&input, &output, i, c, &selected_entry->kernel);
            }

            //start timer and run convolution for each thread
            double start_time = now();
            for (int i = 0; i < c; i++)
            {
                pthread_create(&threads[i], NULL, convolve, boxes[i]);
            }
            for (int i = 0; i < c; i++)
            {
                pthread_join(threads[i], NULL);
                free(boxes[i]);//free box memory
            }
            if(c > 1)
                printf("With %d cores, took %5.3f seconds\n", c, now()-start_time);
            else
                printf("With 1 core, took %5.3f seconds\n", now()-start_time);
        }
        encode_and_store(&output, output_file_name);
    }
    /**
     * If the user has not decided to run as sequentially, program runs
     * one convolution on p processors
     */
    else
    {
        for (int i = 0; i < num_threads_used; i++)
        {
            boxes[i] = create_mystery_box(&input, &output, i, num_threads_used, &selected_entry->kernel);
        }
        double start_time = now();
        for (int i = 0; i < num_threads_used; i++)
        {
            pthread_create(&threads[i], NULL, convolve, boxes[i]);
        }
        for (int i = 0; i < num_threads_used; i++)
        {
            pthread_join(threads[i], NULL);
            free(boxes[i]); //free box memory
        }
        if(num_threads_used > 1)
                printf("With %d cores, took %5.3f seconds\n", num_threads_used, now()-start_time);
            else
                printf("With 1 core, took %5.3f seconds\n", now()-start_time);
        encode_and_store(&output, output_file_name);
    }

    //free images
    free_image(&input);
    free_image(&output);
}
