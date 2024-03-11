#!/bin/bash

# Define the list of remove_special_tokens values
remove_special_tokens=("True" "False")

# Iterate through the files in the directory
for file in /scratch/project_2005092/Anni/qa-register/models_for_Amanda/*; do
    if [[ $file != "/scratch/project_2005092/Anni/qa-register/models_for_Amanda/en_annotated" ]]; then
        # Extract the file name
        file_name=$(basename "$file")
        for remove_tokens_value in "${remove_special_tokens[@]}"
        do
            # Execute the Python program with the current parameters
            python qa_detection.py --model_path "$file_name" --input_lang detect --remove_special_tokens "$remove_tokens_value" --label_start_only True
        done
    fi
done
