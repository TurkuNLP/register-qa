#!/bin/bash
#SBATCH --job-name=qa_new
#SBATCH --account=project_2005092 #2005092 # 2000539
#SBATCH --partition=gpu #gputest
#SBATCH --time=00:20:00 #2ish hours for one epoch with large, 30min with base
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G  # need a lot of memory if mapping from scratch ... # now maybe not so much
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err

module load pytorch 




# GRID SEARCH REQUIRES THE OUTER SCRIPT THAT DOES SBATCH FOR EVERYTHING

# EPOCHS=10 ##{5..7..1} # last one is increment
# LR="4e-6 5e-6 7e-5 8e-6" # learning rate
# BATCH=8 #"7 8"
# # # $I={1..2..1} also loop so that it reproduces the same thing a couple of times?
# MODEL="xlm-roberta-base"

# # for BATCH in $BATCH; do
# # for EPOCHS in $EPOCHS_; do
# for rate in $LR; do
# echo "learning rate: $rate batch: $BATCH epochs: $EPOCHS"
# echo "START: $(date)"
# srun python3 training/qa-multiclass.py \
#     --model $MODEL \
#     --train_set data/CORE-corpus/train.tsv.gz data/FinCORE_full/train.tsv data/SweCORE/swe_train.tsv data/FreCORE/fre_train.tsv  \
#     --dev_set data/CORE-corpus/dev.tsv.gz data/FinCORE_full/dev.tsv data/SweCORE/swe_dev.tsv data/FreCORE/fre_dev.tsv  \
#     --test_set data/CORE-corpus/test.tsv.gz data/FinCORE_full/test.tsv data/SweCORE/swe_test.tsv data/FreCORE/fre_test.tsv \
#     --save_name "register-qa-binary-$rate" \
#     --batch $BATCH --epochs $EPOCHS --learning $rate --save --weights

# echo "END: $(date)"
# done
# done
# done

echo "START: $(date)"

EPOCHS=10 #3
LR=4e-6 # "1e-5 4e-6 5e-6 7e-5 8e-6"
BATCH=8
MODEL="xlm-roberta-base" # or large

#echo "learning rate: $LR treshold: $TR batch: $BATCH epochs: $EPOCHS"

#data/CORE-corpus/train.tsv.gz data/FinCORE_full/train.tsv data/SweCORE/swe_train.tsv data/FreCORE/fre_train.tsv 
#data/CORE-corpus/dev.tsv.gz data/FinCORE_full/dev.tsv data/SweCORE/swe_dev.tsv data/FreCORE/fre_dev.tsv 
#data/CORE-corpus/test.tsv.gz data/FinCORE_full/test.tsv data/SweCORE/swe_test.tsv data/FreCORE/fre_test.tsv
srun python3 training/qa-multiclass.py \
    --model $MODEL \
    --train_set data/CORE-corpus/train.tsv.gz data/FinCORE_full/train.tsv data/SweCORE/swe_train.tsv data/FreCORE/fre_train.tsv  \
    --dev_set data/CORE-corpus/dev.tsv.gz data/FinCORE_full/dev.tsv data/SweCORE/swe_dev.tsv data/FreCORE/fre_dev.tsv  \
    --test_set data/CORE-corpus/test.tsv.gz data/FinCORE_full/test.tsv data/SweCORE/swe_test.tsv data/FreCORE/fre_test.tsv \
    --save_name "register-qa-binary-$LR" \
    --batch $BATCH --epochs $EPOCHS --learning $LR --save --weights


echo "END: $(date)"