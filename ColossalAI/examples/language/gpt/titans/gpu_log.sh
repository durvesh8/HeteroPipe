#!/bin/bash

# Path to the CSV file
csv_file_path="${exp_name}_gpu_info.csv"

# Add headers to the CSV file
echo "GPU_ID,GPU_Usage,Memory_Used,Temperature,Power_Draw" > $csv_file_path

# Run nvidia-smi in a loop
while true; do
    # Get the GPU ID, usage, memory, temperature, and power consumption
    STATS=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader)

    # Add the stats to the CSV file
    echo "$STATS" >> $csv_file_path

    # Sleep for a while
    sleep 5
done
