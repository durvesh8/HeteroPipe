#!/bin/bash


# Pull changes
git pull

# Navigate to the directory
cd ~/work/HeteroPipe/ColossalAI-0.2.0

# Export paths
export PATH="/usr/local/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"

# Run pip install
pip install .

cd ~/work/HeteroPipe/ColossalAI/examples/language/gpt/titans/
