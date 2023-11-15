#!/bin/bash
#SBATCH --job-name=qa_register
#SBATCH --account=project_2005092
#SBATCH --partition=gpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs/%j.out
#SBATCH --error=../logs/%j.err


module purge
module load pytorch

echo "START: $(date)"

srun python3 predict.py --model "models/new_model2" --data "data/FinCORE_full/test.tsv" --tokenizer "xlm-roberta-base" --filename "data/predictions/qa_xlmr.tsv"

echo "END: $(date)"