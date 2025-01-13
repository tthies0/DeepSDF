#!/bin/bash

sudo apt install build-essential cmake libgl1-mesa-dev libglew-dev 
[ -f ./third-party/nanoflann/include/nanoflann.hpp ] && mkdir -p ./third-party/nanoflann/nanoflann && mv ./third-party/nanoflann/include/nanoflann.hpp ./third-party/nanoflann/nanoflann/nanoflann.hpp

ROOT_DIR=$(pwd)
THIRD_PARTY_DIR="$ROOT_DIR/third-party"
DEPENDENCIES=("CL11" "eigen" "Pangolin")

for dependency in "${DEPENDENCIES[@]}"; do
    DEP_DIR="$THIRD_PARTY_DIR/$dependency"
    
    if [ -d "$DEP_DIR" ]; then
        echo "Building and installing $dependency"
        
        # Create or clean the build directory for the dependency
        BUILD_DIR="$DEP_DIR/build"
        if [ -d "$BUILD_DIR" ]; then
            echo "Cleaning existing build directory for $dependency"
            rm -rf "$BUILD_DIR/*"  # Remove all files and subdirectories
        else
            mkdir -p "$BUILD_DIR"
        fi
        
        # Change to the build directory
        cd "$BUILD_DIR"
        
        # Configure, build, and install
        cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
        make -j
        sudo make install
        
        # Return to the root directory
        cd "$ROOT_DIR"
    else
        echo "Warning: Directory for $dependency not found. Make sure to run \"git submodule update --init --recursive\""
    fi
done

if [ -d "./build" ]; then
    echo "Cleaning existing build directory for $dependency"
    rm -rf "./build/*"  # Remove all files and subdirectories
else
    mkdir -p "$BUILD_DIR"
fi

cd ./build

cmake .. && make -j

echo "Specified dependencies built and installed."