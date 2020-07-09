# Math Library Optimizations
Repository for testing and analyzing the performance of low-level math libraries in RHEL

# Current Code

`src/libgomp`: Contains C code used for testing the performance of libgomp

# Compiling the Code

Under each subdirectory (e.g., `src/libgomp`), there is a Makefile. To build the code,

```
$ make -C src/<library>
```

To clean the code,

```
$ make -C src/<library> clean
```

Replace `<library>` with an existing library -- e.g., `libgomp`
