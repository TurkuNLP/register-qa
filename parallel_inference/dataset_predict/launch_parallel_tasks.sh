#!/bin/bash
set -e
#export CACHE_DIR=/scratch/project_2005092/Anni/new_cache_dir/
export CACHE_DIR=/scratch/project_462000185/anni/new_cache_dir/

#### Change if required
TOKENIZER_PATH="xlm-roberta-base"
MODEL_PATH="./../../models/register-qa-binary-4e-6" 
####1

# now from 1M to 10M with 1M increase -> 10 batch jobs?
for start in {1000000..2000000..1000000};do 

    SAVE_NAME="dataset-register-labels-qa-$start.tsv"
    echo "Launching processing for dataset $start \n with save file path: $SAVE_NAME "
    # -J {$SAVE_NAME}
    sbatch predict_multi_gpu.sh \
        --batch_size_per_device 64 \
        --output $SAVE_NAME \
        --dataset 'tiiuae/falcon-refinedweb' \
        --start $start \
        --text_name 'content' \
        --id_col_name 'url' \
        --labels 'qa' \
        --tokenizer_path $TOKENIZER_PATH \
        --model_path $MODEL_PATH
        #--long_text \

        # for multiclass use qa for labels
        # for qa to make new predictions use tsv and full labels
        # cc-fi, parsebank, suomi24 have "id", mc4 has url or I can also put nothing there for id_col_name

    sleep 0.5 
done