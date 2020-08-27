# Math Library Optimizations
Repository for testing and analyzing the performance of low-level math libraries in RHEL

# Current Code

`src/libgomp`: Contains C code used for testing the performance of libgomp

# Compiling the Code

Under each subdirectory (e.g., `src/libgomp`), there is a Makefile. To build the code,

```bash
$ make -C src/<library>
```

To clean the code,

```bash
$ make -C src/<library> clean
```

Replace `<library>` with an existing library -- e.g., `libgomp`


# Dockerfiles

This repository contains Dockerfiles for running various benchmarks. To build,

```bash
$ podman build -f Dockerfiles/Dockerfile.<name> .
```

The `WORKDIR` value is set to the directory where the benchmarks should be run. You shouldn't need to change directories.

Note that if you're using the custom libgomp podman build, you should run `scripts/setup_gcc_volume.sh` first to prepare the custom libraries for mounting in the container.

Once that is done, you can run the image

```bash
$ podman run -v /tmp/gcc-podman-mnt:/gcc:z ${IMAGE_NAME_OR_ID} /bin/bash
```

Make sure to run `run_me_first.sh` before continuing with anything. (This script is the only script in the current working directory.) This script will prepare the libgomp benchmarks build for you. i.e., you should run the script like so:

```bash
$ . ./run_me_first.sh
```

Running this script will setup your gcc/libgomp build, and it will change you to the appropriate directory for running the benchmarks.

## libgomp

To run libgomp ULHPC benchmarks, 

```
$ ./bin/xhpcg <nx> <ny> <nz> <desired-runtime>
$ #e.g., ./bin/xhpcg 280 280 280 1800
```

The `nx`, `ny`, and `nz` represent the global problem size in the x, y, and z dimensions. Ideally, you want official runtimes to be >1800s. (For more information, [see this pdf](https://www.hpcg-benchmark.org/downloads/sc19/HPCG-AMD-Lulu.pdf) from AMD.)
