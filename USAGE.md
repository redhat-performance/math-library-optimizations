# Usage

## Matrix Multiplication with GOMP

Currently, there is only one piece of code in this repository, src/libgomp/matrixi\_multiplication.c

### Compiling

Compiling is easy. Simply run `make -C src/libgomp`. This command will generate four executables that start with `matmul`. Each executable is given a name based on the byte alignment used (if any).

### Running

The executables all take a minimum of three arguments: `m`, `n`, and `k`. Essentially, we perform matrix multiplcation of matrices `A` and `B`, whereby:

```
A = m x n matrix
B = n x k matrix
```

The matrices are populated with random `double` values. Users do not get the option to control the contents of the matrices.

Sample usage:

```bash
$ OMP_NUM_THREADS=$N ./matmul 2000 2000 2000
```

where `$N` is the number of OpenMP threads to use.

You can also set the number of iterations to use when trying to capture an average performance time. To do so, simply add a 4th argument:

```bash
$ OMP_NUM_THREADS=$N ./matmul_<matmul_version> 2000 2000 2000 $num_iterations
```

Below is an example of running an 8-byte aligned matrix multiplication task a total of 48 times, broken into two 24-thread chunks.

```bash
$ OMP_NUM_THREADS=24 ./matmul_8byte_aligned 256 256 256 48
```

**NOTE:** It is strongly recommended that you set the number of iterations to be a factor of `OMP_NUM_THREADS`. It is also strongly recommended to use matrix dimensions that are powers of two when using alignments.
