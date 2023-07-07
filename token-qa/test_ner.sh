#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=20G 
#SBATCH -p small-g # gpu on puhti
#SBATCH -t 00:10:00 
#SBATCH --gres=gpu:mi250:1 
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000321 #project_2005092 register-labeling on puhti
#SBATCH -o ./../logs/%j.out
#SBATCH -e ./../logs/%j.err


rm -f ./../logs/latest.out ./../logs/latest.err
ln -s $SLURM_JOBID.out ./../logs/latest.out
ln -s $SLURM_JOBID.err ./../logs/latest.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$@"
echo "START $SLURM_JOBID: $(date)"

# if on PUHTI
# module load pytorch

# if on LUMI

module use /appl/local/csc/modulefiles
module load pytorch 


MODEL="../models/token_classification/qa_detection_model"
TOKENIZER="xlm-roberta-base"
TEXT="../data/splits/labelled/binary/sorted_qa_binary_parsebank.tsv"
BATCH=64
OUTPUT="ner_test.tsv"

srun python3 test_qa_detection.py --model_path $MODEL --batch $BATCH --tokenizer_path $TOKENIZER --batch_size_per_device $BATCH --output $OUTPUT --text $TEXT

echo "END $SLURM_JOBID: $(date)"
