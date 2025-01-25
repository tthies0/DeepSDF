#!/bin/bash

# Define the target directory where all files will be saved
TARGET_DIR="./target"

# Ensure the target directory exists
mkdir -p "$TARGET_DIR"

# Find all "sdf.npz" files in subdirectories
find /home/daniel/Documents/DeepSDF/data/testdata/SdfSamples/ShapeNetV2/04256520 -type f -name "sdf.npz" | while read -r file; do
  # Get the parent directory name
  parent_dir=$(basename "$(dirname "$file")")
  
  # Copy the file to the target directory with the new name
  mv "$file" "$TARGET_DIR/${parent_dir}.npz"
done

echo "All files have been copied and renamed to $TARGET_DIR"
