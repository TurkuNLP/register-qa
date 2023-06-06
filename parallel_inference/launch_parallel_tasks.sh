#!/bin/bash
set -e
export CACHE_DIR=/scratch/project_462000185/risto/huggingface-t5-checkpoints/cache_dir/

### Split the dataset into chunks of ~4G with 
# split -C 4G <dataset-name> --additional-suffix .jsonl

#### Change if required
TOKENIZER_PATH="TurkuNLP/bert-base-finnish-cased-v1"
DATA_DIR="/path/to/data/splits"
####

for file in $DATA_DIR/x*.jsonl;do
    # SAVE_NAME=$(basename $file)"register-labels.tsv"
    SAVE_NAME="$file-register-labels.tsv"
    echo "Launching processing for: $file \n with save file path: $SAVE_NAME "
    sbatch -J "$(basename $SAVE_NAME)" predict_multi_gpu.sh \
        --tokenizer_path $TOKENIZER_PATH \
        --batch_size_per_device 64 \
        --file_type 'jsonl' \
        --text $file \
        --output $SAVE_NAME \
        --id_col_name 'id'

    sleep 0.5 
done