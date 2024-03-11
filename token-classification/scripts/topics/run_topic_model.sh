#!/bin/bash
#SBATCH --job-name=topic-model
#SBATCH --account=project_2005092
#SBATCH --time=02:15:00
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


data=$1
num_topics=$2

source tvenv/bin/activate

srun python3 topic_modelling.py \
  $data \
  -1 \
  $num_topics

seff $SLURM_JOBID
