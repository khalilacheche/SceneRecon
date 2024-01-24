#!/bin/bash


run_name="pf_mv_mesh_sampling_cnn"




# Check if the file exists
if [ ! -f "/scratch/students/2023-fall-acheche/data/scannet/scannet-finerecon/scans/test_orig.txt" ]; then
    echo "Error: File 'your_input_file.txt' not found."
    exit 1
fi

# Read the file into an array
mapfile -t scans < "/scratch/students/2023-fall-acheche/data/scannet/scannet-finerecon/scans/test_orig.txt"

# Iterate through the elements and execute a command
for scan in "${scans[@]}"; do
    # Replace "your_command" with the actual command you want to execute
    your_command="source activate /scratch/students/2023-fall-acheche/conda/scenerecon && cd /scratch/students/2023-fall-acheche/SceneRecon && python main.py --run_name $run_name --task predict --scene $scan"

    # Execute the command
    eval "$your_command"

    # Check the exit status of the command
    if [ $? -eq 0 ]; then
        echo "Command executed successfully."
    else
        echo "Error executing command for element: $scan"
    fi
done