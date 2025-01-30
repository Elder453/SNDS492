#!/bin/bash
set -e  # Exit immediately if any command fails

# Convert Jupyter notebook to a Python script
jupyter nbconvert --to python research/fish_3D.ipynb

# Concatenate the Python files
cat kuramoto/gkm_gpu.py research/fish_3D.py > combined_output.py

echo "Successfully combined files into combined_output.py"