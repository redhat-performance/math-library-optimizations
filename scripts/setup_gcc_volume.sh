#!/bin/bash

# This script sets up the mount for gcc libs

# Get prefix to search under for gcc related libs, headers...
PREFIX=$1

# Copies gcc libs to the mount dir
copy_libs_standard_rhel () {
    for lib in $1; do
	
        # Get the path to the slib name
        lib_path=$(echo $lib | rev | cut -d"/" -f2- | rev)

        # Create a proposed path for where we will store the lib
        proposed_lib_path=${gcc_mount_dir}${lib_path}

        # Check if dir exists
        if [[ ! -d ${proposed_lib_path} ]]; then
            mkdir -p ${proposed_lib_path}
        fi

        # Copy libs
        cp -n ${lib} ${gcc_mount_dir}${lib}

    done
}

# Copies gcc headers to the mount dir
copy_headers_standard_rhel () {
    for header in $1; do
	
        # Check if the headers are file or folder names
        header_path=$(echo $header | rev | cut -d"/" -f2- | rev)

        # Create a proposed path for where we will store the header
        proposed_header_path=${gcc_mount_dir}${header_path}

        # Check if dir exists
        if [[ ! -d ${proposed_header_path} ]]; then
            mkdir -p ${proposed_header_path}
        fi

        # Copy headers
        cp -n -R ${header} ${gcc_mount_dir}${header}

    done
}

# Copies gcc executables to the mount dir
copy_executables_standard_rhel () {
    for executable in $1; do
	
        # Get the path to the slib name
        executable_path=$(echo $executable | rev | cut -d"/" -f2- | rev)

        # Create a proposed path for where we will store the executable
        proposed_executable_path=${gcc_mount_dir}${executable_path}

        # Check if dir exists
        if [[ ! -d ${proposed_executable_path} ]]; then
            mkdir -p ${proposed_executable_path}
        fi

        # Copy executables
        cp -n ${executable} ${gcc_mount_dir}${executable}

    done
}

# Copies libexec files to the mount dir
copy_libexec_standard_rhel () {
    for executable in $1; do
	
        # Get the path to the slib name
        libexec_path=$(echo $executable | rev | cut -d"/" -f2- | rev)

        # Create a proposed path for where we will store the executable
        proposed_libexec_path=${gcc_mount_dir}${libexec_path}

        # Check if dir exists
        if [[ ! -d ${proposed_libexec_path} ]]; then
            mkdir -p ${proposed_libexec_path}
        fi

	# Copy libexec executable(s)
        cp -R -n ${executable} ${gcc_mount_dir}${executable}

    done
}

# Copies conf files to the mount dir
copy_configs_standard_rhel () {
    for config_file_or_dir in $1; do
	
        # Get the path to the config file or dir
        conf_path=$(echo $conf_file_or_dir | rev | cut -d"/" -f2- | rev)

        # Create a proposed path for where we will store the executable
        proposed_conf_path=${gcc_mount_dir}${conf_path}

        # Check if dir exists
        if [[ ! -d ${proposed_conf_path} ]]; then
            mkdir -p ${proposed_conf_path}
        fi

	# Copy configs (dirs or files)
        cp -R -n ${config_file_or_dir} ${gcc_mount_dir}${conf_file_or_dir}

    done
}

# Copies conf files to the mount dir
copy_specs_standard_rhel () {
    for spec_file in $1; do
	
        # Get the path to the config file or dir
        spec_path=$(echo $spec_file | rev | cut -d"/" -f2- | rev)

        # Create a proposed path for where we will store the executable
        proposed_spec_path=${gcc_mount_dir}${spec_path}

        # Check if dir exists
        if [[ ! -d ${proposed_spec_path} ]]; then
            mkdir -p ${proposed_spec_path}
        fi

	# Copy *.spec files
        cp -R -n ${spec_file} ${gcc_mount_dir}${spec_file}

    done
}

# Greps for relevant gcc* file names (e.g,. *.so, *.a, *.h, *.c, etc.)
get_file_names () {

    # List of package files
    pkg_files=("$@")

    # Initialize results arrays
    shared_libs=()
    static_libs=()
    configs=()
    headers=()
    executables=()
    libexec_executables=()
    specs=()

    for current_file in "${pkg_files[@]}"; do

	if [[ "$current_file" == */share/* ]] || [[ "$current_file" == */licenses* ]]; then
            continue

        elif [[ ! -z `echo $current_file | grep "\/bin\/"` ]] || [[ ! -z `echo $current_file | grep "\/sbin\/"` ]]; then
	    executables+=($current_file)
        
	elif [[ "$current_file" == *conf* ]]; then
            configs+=($current_file)

	elif [[ "$current_file" == *.so* ]] && [[ "$current_file" != *py* ]]; then
            shared_libs+=($current_file)
        
        elif [[ "$current_file" == *.o ]]; then
	    shared_libs+=($current_file)

        elif [[ ${current_file: -2} == ".a" ]]; then
            static_libs+=($current_file)
        
	elif [[ ${current_file: -2} == ".h" ]] || [[ $current_file == *include* ]]; then
	    headers+=($current_file)

        elif [[ ! -z `echo $current_file | grep "\/libexec"` ]]; then
	    libexec_executables+=($current_file)

        elif [[ "$current_file" == *.spec ]]; then
	    specs+=($current_file)

	fi
    done

    # Print individual arrays, separated by '# '
    echo ${shared_libs[@]}
    echo "# "
    echo ${static_libs[@]}
    echo "# "
    echo ${headers[@]}
    echo "# "
    echo ${executables[@]}
    echo "# "
    echo ${libexec_executables[@]}
    echo "# "
    echo ${configs[@]}
    echo "# "
    echo ${specs[@]}

}


# Define the mount name
GCC_MOUNT_NAME="gcc-podman-mnt"

# Setup where the mount will go
gcc_mount_dir="/tmp/${GCC_MOUNT_NAME}"
if [[ -d ${gcc_mount_dir} ]]; then
    rm -rf ${gcc_mount_dir}
fi
mkdir -p ${gcc_mount_dir}

# If we're using a standard RHEL box, let's copy
if [[ -z ${PREFIX} ]] || [[ ${PREFIX} == /usr* ]]; then

    # Logic taken and modified from https://serverfault.com/a/429163
    provider_pkgs=$(yum deplist gcc-c++ pkgconf libpkgconf glibc-devel glibc-headers isl libgomp | awk '/x86_64/,/provider/ {print $2}')
    provider_pkgs+=" gcc-c++ glibc-devel binutils glibc-common"

    # Get unique packages
    unique_provider_pkgs=$(echo "$provider_pkgs" | xargs -n1 | sort -u | xargs)

    for pkg in $unique_provider_pkgs; do

	# Print out unique pkg
	echo "----------------------"
	echo "$pkg"
	echo "----------------------"

	# Get output from repoquery
        repoquery_output=$(yum repoquery -q -l $pkg)

	# Convert repoquery output to an array
	repoquery_arr=($repoquery_output)

	# Get relevant files
        relevant_files=`get_file_names "${repoquery_arr[@]}"`

	shared_libs=`echo $relevant_files | cut -d '#' -f1`
	static_libs=`echo $relevant_files | cut -d '#' -f2`
	headers=`echo $relevant_files | cut -d '#' -f3`
	executables=`echo $relevant_files | cut -d '#' -f4`
	libexec_executables=`echo $relevant_files | cut -d '#' -f5`
	configs=`echo $relevant_files | cut -d '#' -f6`
	specs=`echo $relevant_files | cut -d '#' -f7`

        copy_libs_standard_rhel "${shared_libs[@]}"
	copy_libs_standard_rhel "${static_libs[@]}"
	copy_headers_standard_rhel "${headers[@]}"
	copy_executables_standard_rhel "${executables[@]}"
        copy_libexec_standard_rhel "${libexec_executables[@]}"
        copy_configs_standard_rhel "${configs[@]}"
        copy_specs_standard_rhel "${specs[@]}"
	
    done

else

    # We're not using standard RHEL or we're using a custom install location via
    # --installroot=/some/path, so let's use the prefix to copy data
    if [[ -d ${PREFIX} ]]; then
	echo "Could not find path '${PREFIX}'"
	exit 1
    fi

    # Check if the install path has the appropriate /usr, /lib, etc. folders
    required_folders="/usr /lib /lib64 /bin /libexec"
    for folder in ${required_folders}; do
        if [[ -d ${PREFIX}${folder} ]]; then
            echo "Directory not setup properly. Missing folder: ${PREFIX}${folder}"
	fi
    done

    # Now copy appropriate files
    for folder in ${required_folders}; do
	cp -R -n ${PREFIX}${folder} ${gcc_mount_dir}
    done

fi
