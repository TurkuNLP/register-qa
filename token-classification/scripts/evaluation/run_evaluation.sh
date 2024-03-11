#!/bin/bash

# Define the list of remove_special_tokens values
IAA=("False")

# Iterate through the files in the directory
for file in /scratch/project_2002026/amanda/register-qa/evaluation/final_3_results/predictions/*; do
    
    # Extract the file name
    file_name=$(basename "$file")
    #for IAA_value in "${IAA[@]}"
    #do
        # Execute the Python program with the current parameters
    python evaluate.py --input "$file_name"  # --IAA $IAA_value 
    #done
    
done