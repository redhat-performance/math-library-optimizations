#!/bin/bash

# Setup vars
TUTORIALS_GIT_PATH="/opt/ULHPC"
TUTORIALS_DEDICATED_DIR="${HOME}/tutorials/HPL"
REFERENCE_DIR="ref.ulhpc.d"

# Get current dir
LIBGOMP_DIR=`pwd`/..

# Clone the git repo
if [[ -d ${TUTORIALS_GIT_PATH} ]]; then
    echo "Removing existing ULHPC/tutorials git path"
    rm -rf ${TUTORIALS_GIT_PATH}
fi
mkdir -p ${TUTORIALS_GIT_PATH}
cd ${TUTORIALS_GIT_PATH}
git clone https://github.com/ULHPC/tutorials.git

# Enter the 'tutorials' git repo
cd tutorials

# Build it
make setup

# Prepare a dedicated directory to work on the tutorial
if [[ -d ${TUTORIALS_DEDICATED_DIR} ]]; then
    echo "Removing existing tutorials dedicated dir..."
    rm -rf ${TUTORIALS_DEDICATED_DIR}
fi
mkdir -p ${TUTORIALS_DEDICATED_DIR}

# Create ref.ulhpc.d
cd ${TUTORIALS_DEDICATED_DIR}

# Get HPCG benchmark and untar it
wget http://www.hpcg-benchmark.org/downloads/hpcg-3.0.tar.gz
tar xvzf hpcg-3.0.tar.gz

# Enter the dir
cd hpcg-3.0

# Setup
mkdir build
cd build
../configure GCC_OMP
make
