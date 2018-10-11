#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "mpi.h"

/* Return the current time. */
double
now(void)
{
#define ONE_BILLION (double)1000000000.0
  struct timespec current_time;
  clock_gettime(CLOCK_REALTIME, &current_time);
  return current_time.tv_sec + (current_time.tv_nsec / ONE_BILLION);
}

double start_time;

void
start_timer(void) {
  start_time = now();
}

double
get_timer(void) {
  return now() - start_time;
}

void
broad_barrier(int rank, int procs)
{
  long int random_value;
  int broadcaster_rank = procs - 1;

  if (rank == broadcaster_rank) {
    srandom(time(NULL) + rank);
    random_value = random() / (RAND_MAX / 10);
    printf("%d: broadcasting %ld\n", rank, random_value);
  }

  MPI_Bcast((void *)&random_value,
            1, MPI_LONG,
            broadcaster_rank,
            MPI_COMM_WORLD);

  if (rank != broadcaster_rank) {
    printf("%d: received %ld\n", rank, random_value);
  }

  int nap_time = random_value + (2 * rank);
  printf("%d @ %0.2f: sleeping %ds\n",
         rank, get_timer(), nap_time);
  sleep(nap_time);

  printf("%d @ %0.2f: enter b-a-r-r-i-e-r\n",
         rank, get_timer());
  //MPI_Barrier(MPI_COMM_WORLD);
  printf("%d @ %0.2f: leave barrier\n",
         rank, get_timer());
}

int
main (int argc, char **argv)
{
  int num_procs;
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  start_timer();

  printf("%d: hello (p=%d)\n", rank, num_procs);
  broad_barrier(rank, num_procs);
  printf("%d: goodbye\n", rank);

  MPI_Finalize();
}
