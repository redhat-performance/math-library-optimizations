CC=gcc
CFLAGS=-flto -fopenmp -lgomp -fomit-frame-pointer

all: matrix_multiplication.c
	$(CC) $(CFLAGS) -o matmul matrix_multiplication.c

clean: 
	/usr/bin/rm -f matmul
