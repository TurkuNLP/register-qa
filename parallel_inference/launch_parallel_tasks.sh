#!/bin/bash
set -e
#export CACHE_DIR=/scratch/project_2005092/Anni/new_cache_dir/
export CACHE_DIR=/scratch/project_462000185/anni/new_cache_dir/

### Split the dataset into chunks of ~4G with 
# split -C 4G <dataset-name> --additional-suffix .jsonl

#### Change if required
#TOKENIZER_PATH="TurkuNLP/bert-base-finnish-cased-v1"
#MODEL_PATH = "finbert-base-fin-0.00002-MTv2.pt"
DATA_DIR="./../data/splits" # anni/data/splits
####

# do this first for just one label!
for file in $DATA_DIR/parsebank_00*.jsonl;do 
    # SAVE_NAME=$(basename $file)"register-labels.tsv"
    SAVE_NAME="$file-register-labels.tsv"
    echo "Launching processing for: $file \n with save file path: $SAVE_NAME "
    sbatch -J "$(basename $SAVE_NAME)" predict_multi_gpu.sh \
        --batch_size_per_device 64 \
        --file_type 'jsonl' \
        --text $file \
        --output $SAVE_NAME \
        --id_col_name 'id' \
        --labels 'upper' \
        --long_text \
        #--tokenizer_path $TOKENIZER_PATH \
        #--model_path $MODEL_PATH

        # cc-fi, parsebank have "id", mc4 has url or I can also put nothing there

    sleep 0.5 
done