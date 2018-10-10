#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"

void
round_robin(int rank, int procs)
{
  long int rand_mine, rand_prev;
  int rank_next = (rank + 1) % procs;
  int rank_prev = rank == 0 ? procs - 1 : rank - 1;
  MPI_Status status;
  MPI_Request send_request;

  srandom(time(NULL) + rank);
  rand_mine = random() / (RAND_MAX / 100);
  printf("%d: random is %ld\n", rank, rand_mine);

  printf("%d: sending %ld to %d\n",
         rank, rand_mine, rank_next);
  MPI_Isend((void *)&rand_mine, 1, MPI_LONG,
            rank_next, 1, MPI_COMM_WORLD,
            &send_request);
  MPI_Recv((void *)&rand_prev, 1, MPI_LONG,
           rank_prev, 1, MPI_COMM_WORLD,
           &status);

  printf("%d: I had %ld, %d had %ld\n",
         rank, rand_mine,
         rank_prev, rand_prev);

  int wait_count = 0;
  int flag = 0;
  do {
    wait_count++;
    MPI_Test(&send_request, &flag, &status);
  } while(!flag);
  printf("%d: MPI_Test calls: %d\n", rank, wait_count);
}

int
main (int argc, char **argv)
{
  int num_procs;
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  printf("%d: hello (p=%d)\n", rank, num_procs);
  round_robin(rank, num_procs);
  printf("%d: goodbye\n", rank);

  MPI_Finalize();
}
