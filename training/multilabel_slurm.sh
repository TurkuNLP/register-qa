#!/bin/bash
#SBATCH --job-name=translated
#SBATCH --account=project_2005092 #2005092 # 2000539
#SBATCH --partition=gpu #gputest
#SBATCH --time=02:30:00 #2ish hours for one epoch with large, 30min with base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5 # from 10 to 1
#SBATCH --mem-per-cpu=2G # 8000 # need a lot of memory if mapping from scratch ...
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 

# # with current settings set time to 16 x 1.30 = 24

# EPOCHS=5 ##{5..7..1} # last one is increment
# LR="4e-6 5e-6 7e-5 8e-6" # learning rate
# BATCH=8 #"7 8"
# TR="0.3 0.4 0.5 0.6" #treshold
# # $I={1..2..1} also loop so that it reproduces the same thing a couple of times?


# TRAIN=main_labels_only/ja_FINAL.modified.tsv.gz
# TEST=test_sets/jpn_test_modified.tsv

# echo "train: $TRAIN test: $TEST"

# # for BATCH in $BATCH; do
# # for EPOCHS in $EPOCHS_; do
# for rate in $LR; do
# for treshold in $TR; do
# echo "learning rate: $rate treshold: $treshold batch: $BATCH epochs: $EPOCHS"
# srun python3 register-multilabel.py \
#     --train_set $TRAIN \
#     --test_set $TEST \
#     --batch $BATCH \
#     --treshold $treshold \
#     --epochs $EPOCHS \
#     --learning $rate
# done
# done
# done
# done


EPOCHS=3 #5
LR=8e-6 # "1e-5 4e-6 5e-6 7e-5 8e-6"
TR=0.5    # "0.3 0.4 0.5 0.6"
BATCH=8
MODEL="xlm-roberta-base" # or large

echo "learning rate: $LR treshold: $TR batch: $BATCH epochs: $EPOCHS"

#data/CORE-corpus/train.tsv.gz data/FinCORE_full/train.tsv data/SweCORE/swe_train.tsv data/FreCORE/fre_train.tsv 
#data/CORE-corpus/dev.tsv.gz data/FinCORE_full/dev.tsv data/SweCORE/swe_dev.tsv data/FreCORE/fre_dev.tsv 
#data/CORE-corpus/test.tsv.gz data/FinCORE_full/test.tsv data/SweCORE/swe_test.tsv data/FreCORE/fre_test.tsv
srun python3 training/register-multilabel.py \
    --model $MODEL \
    --train_set data/CORE-corpus/train.tsv.gz data/FinCORE_full/train.tsv data/SweCORE/swe_train.tsv data/FreCORE/fre_train.tsv \
    --dev_set data/CORE-corpus/dev.tsv.gz data/FinCORE_full/dev.tsv data/SweCORE/swe_dev.tsv data/FreCORE/fre_dev.tsv \
    --test_set data/CORE-corpus/test.tsv.gz data/FinCORE_full/test.tsv data/SweCORE/swe_test.tsv data/FreCORE/fre_test.tsv \
    --batch $BATCH --threshold $TR --epochs $EPOCHS --learning $LR --save --weights 


echo "END: $(date)"