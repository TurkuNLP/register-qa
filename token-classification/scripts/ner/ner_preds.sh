#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=20G 
#SBATCH -p small-g # gpu on puhti
#SBATCH -t 01:30:00  # 4 hrs for normal english model
#SBATCH --gres=gpu:mi250:1 
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000321 #project_2005092 register-labeling on puhti
#SBATCH -o ./../../logs/token-qa/%j.out
#SBATCH -e ./../../logs/token-qa/%j.err




rm -f ./../../logs/token-qa/latest.out ./../../logs/token-qa/latest.err
ln -s $SLURM_JOBID.out ./../../logs/token-qa/latest.out
ln -s $SLURM_JOBID.err ./../../logs/token-qa/latest.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$@"
echo "START $SLURM_JOBID: $(date)"

# if on PUHTI
# module load pytorch

# if on LUMI

module use /appl/local/csc/modulefiles
module load pytorch
# first load these and then install the required with pip3 install --user

#module load cray-python # I pip installed the needed packages (pytorch, datasets, transformers, seqeval, evaluate) while this was loaded and now they all work! maybe a stupid way but hey I got them working
# gpu did not work, had to get the roc pytorch version because of amd

# module load CrayEnv

# For Finnish "../../models_for_Amanda/enfigpt_fi_1e-5"
# For English "../../models_for_Amanda/en_annotated/figpt_en-ann_3e-5"
MODEL="../../models_for_Amanda/enfigpt_fi_1e-5"
# ../../data/splits/labelled/binary/sorted_qa_binary_cc-fi_all.tsv
# ../../data/splits/labelled/binary/sorted_qa_binary_mc4-fi.tsv
#../../data/splits/labelled/binary/sorted_qa_binary_parsebank.tsv
#../../data/splits/labelled/binary/sorted_falcon8M.tsv
DATA="../../data/splits/labelled/binary/sorted_qa_binary_parsebank.tsv" 
SAVE="../../data/qa_results/final/parsebank_averageaggregation.jsonl"

srun python3 ner_preds.py --model $MODEL --data $DATA --save $SAVE 

echo "END $SLURM_JOBID: $(date)"
