#!/bin/bash
set -e
#export CACHE_DIR=/scratch/project_2005092/Anni/new_cache_dir/
export CACHE_DIR=/scratch/project_462000185/anni/new_cache_dir/

### Split the dataset into chunks of ~4G with 
# split -C 4G <dataset-name> --additional-suffix .jsonl

#### Change if required
TOKENIZER_PATH="xlm-roberta-large" #"TurkuNLP/bert-base-finnish-cased-v1" #"xlm-roberta-base" for qa
MODEL_PATH=./../models/register-qa-binary-4e-6 #"./../models/xlmr-large-en02-fin-fr_mt-sv_mt-0.000006-MT.pt" # "" for qa
DATA_DIR="./../data/splits/og" # anni/data/splits
####1

# do this first for just one label!
for file in $DATA_DIR/suomi24_*;do #suomi24* #labelled/sorted_qa_parsebank.tsv
    # SAVE_NAME=$(basename $file)"register-labels.tsv"
    SAVE_NAME="$file-register-labels.tsv" #have to move the results to another folder
    echo "Launching processing for: $file \n with save file path: $SAVE_NAME "
    sbatch -J "$(basename $SAVE_NAME)" predict_multi_gpu.sh \
        --batch_size_per_device 64 \
        --file_type 'jsonl' \
        --text $file \
        --output $SAVE_NAME \
        --id_col_name 'id' \
        --labels 'qa' \
        --tokenizer_path $TOKENIZER_PATH \
        --model_path $MODEL_PATH
        #--long_text \

        # for multiclass use qa for labels
        # for qa to make new predictions use tsv and full labels
        # cc-fi, parsebank, suomi24 have "id", mc4 has url or I can also put nothing there for id_col_name

    sleep 0.5 
done