#makefile

CC=mpicc
CFLAGS=-Wall
EXEC=mpiexec
RM=rm

dmat: mat-mult.o
	$(CC)

exec: $(EXEC) -np 4 ./a.out -a a.txt -b b.txt -o c.txt -m 3 -n 5 -p 3

.PHONY: clean
clean: rm dmat a.out *.o
