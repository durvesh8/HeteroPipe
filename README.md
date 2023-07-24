# Setting Up the ColossalAI Codebase

This guide will walk you through the process of setting up the ColossalAI codebase on your system.

## Prerequisites

- Conda
- CUDA 11.7
- Git
- Pip

## Steps

1. **Create a new conda environment with Python 3.9**

    Open a terminal and run the following commands:

    ```bash
    conda create -n colossal_new python=3.9
    conda activate colossal_new
    ```

2. **Set up CUDA 11.7 as the main CUDA**

    If you don't have CUDA 11.7 installed, you can download and install it with the following commands:

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
    sudo sh cuda_11.7.0_515.43.04_linux.run
    ```
	Accept EULA and tick only CUDA toolkit, select NO as for symlink already exists.
    Then, add CUDA 11.7 to your PATH and LD_LIBRARY_PATH with these commands:

    ```bash
    export PATH="/usr/local/cuda-11.7/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
    ```

3. **Clone the ColossalAI repository**

    You can clone the ColossalAI repository with this command:

    ```bash
    git clone https://github.com/hpcaitech/ColossalAI.git
    ```

4. **Install the ColossalAI package**

    Navigate to the ColossalAI directory and install the package with this command:

    ```bash
    cd ColossalAI
    pip install .
    ```

5. **Install additional Python packages**

    You can install the additional required Python packages with these commands:

    ```bash
    pip install transformers
    pip install titans
    ```

