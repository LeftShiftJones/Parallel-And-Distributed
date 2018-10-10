#makefile

CC=mpicc
CFLAGS=-Wall
#EXEC=mpiexec
TARGET = mat-mult

all: $(TARGET)

$(TARGET): $(TARGET).c
    $(CC) $(CFLAGS) -o $(TARGET) $(TARGET).c

#exec: $(EXEC) -np 4 ./a.out -a a.txt -b b.txt -o c.txt -m 3 -n 5 -p 3

.PHONY: clean
clean: 
    $(RM) $(TARGET)
