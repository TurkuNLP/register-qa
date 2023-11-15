#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=20G 
#SBATCH -p small-g # gpu on puhti
#SBATCH -t 04:00:00 
#SBATCH --gres=gpu:mi250:1 
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000321 #project_2005092 register-labeling on puhti
#SBATCH -o ./../../logs/%j.out
#SBATCH -e ./../../logs/%j.err


rm -f ./../../logs/latest.out ./../../logs/latest.err
ln -s $SLURM_JOBID.out ./../../logs/latest.out
ln -s $SLURM_JOBID.err ./../../logs/latest.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$@"
echo "START $SLURM_JOBID: $(date)"

# if on PUHTI
# module load pytorch

# if on LUMI

module use /appl/local/csc/modulefiles
module load pytorch # now here is also seqeval, evaluate

#module load cray-python # I pip installed the needed packages (pytorch, datasets, transformers, seqeval, evaluate) while this was loaded and now they all work! maybe a stupid way but hey I got them working
# gpu did not work, had to get the roc pytorch version because of amd

# module load CrayEnv

MODEL="xlm-roberta-base"
TRAIN="../../data/qa_token_classification/lfqa-ready-train.jsonl"
TEST="../../data/qa_token_classification/lfqa-ready-test.jsonl"
DEV="../../data/qa_token_classification/lfqa-ready-validation.jsonl"
BATCH=8
EPOCHS=2
LR=8e-6
SAVE="../../models/token_classification/joined_model_simplified"

srun python3 train_qa_token_classifier.py --model_name $MODEL --train $TRAIN --test $TEST --dev $DEV --batch $BATCH --epochs $EPOCHS --lr $LR --save $SAVE

echo "END $SLURM_JOBID: $(date)"
