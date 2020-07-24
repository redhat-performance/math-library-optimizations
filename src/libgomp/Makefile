CC=gcc
CFLAGS=-g -flto -fopenmp -fomit-frame-pointer -lm

all: matrix_multiplication.c
	$(CC) $(CFLAGS) -o matmul matrix_multiplication.c

clean: 
	/usr/bin/rm -f matmul
