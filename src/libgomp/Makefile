CC=gcc
CFLAGS=-O2 -g -fomit-frame-pointer -lm
OMPFLAGS=-flto -fopenmp -fomit-frame-pointer -msse2 -mavx -mavx2 -DUSE_OPENMP
CACHE_ALIGNMENT=$(shell getconf LEVEL1_DCACHE_LINESIZE)
PAGE_ALIGNMENT=$(shell getconf PAGE_SIZE)

architecture := $(shell uname -p)

ifeq ($(architecture),x86_64)
OMPFLAGS += -DENABLE_PREFETCH_X86
endif

all: unaligned double_aligned cache_line_aligned page_size_aligned no_omp

unaligned: matrix_multiplication.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DUNALIGNED -DALIGNMENT=1 -o matmul_unaligned matrix_multiplication.c

double_aligned: matrix_multiplication.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DALIGNED -DALIGNMENT=8 -o matmul_8byte_aligned matrix_multiplication.c

cache_line_aligned: matrix_multiplication.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DALIGNED -DCACHE_LINE_ALIGNMENT -DALIGNMENT=$(CACHE_ALIGNMENT) -o matmul_cache_line_aligned matrix_multiplication.c

page_size_aligned: matrix_multiplication.c
	$(CC) $(CFLAGS) $(OMPFLAGS) -DALIGNED -DPAGE_SIZE_ALIGNMENT -DALIGNMENT=$(PAGE_ALIGNMENT) -o matmul_page_size_aligned matrix_multiplication.c

no_omp: matrix_multiplication.c
	$(CC) $(CFLAGS) -DNO_OPENMP -o matmul_no_openmp -DALIGNMENT=1 matrix_multiplication.c

clean: 
	/usr/bin/rm -f matmul*

