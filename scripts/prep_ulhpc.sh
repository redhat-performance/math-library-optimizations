#!/bin/bash

# This script copies the contents of /gcc/usr to /usr and preps for
# the installation of ULHPC

# Copy gcc contents
rsync -a /gcc/usr/ /usr

# Setup and install ulhpc
/bin/bash ${LIBGOMP_SCRIPTS}/download_and_setup_ulhpc.sh

# Change to appropriate working directory
cd ${HOME}/tutorials/HPL/hpcg-3.0/build
