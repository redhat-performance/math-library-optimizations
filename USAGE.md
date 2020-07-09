# Usage

## Matrix Multiplication with GOMP

Currently, there is only one piece of code in this repository, src/libgomp/matrix_multiplication.c

### Compiling

Compiling is easy. Simply run `make -C src/libgomp`. This command will generate an executable called `matmul`

### Running

The code takes in 3 arguments: `m`, `n`, and `k`. Essentially, we perform matrix multiplcation of matrices `A` and `B`, whereby:

```
A = m x n matrix
B = n x k matrix
```

The matrices are populated with random `double` values. Users do not get the option to control the contents of the matrices.

Sample usage:

```bash
$ OMP_NUM_THREADS=N ./matmul 2000 2000 2000
```

where `N` is the number of OpenMP threads to use.
