#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem=20G 
#SBATCH -p small 
#SBATCH -t 01:00:00 
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000321 #project_2005092 register-labeling on puhti
#SBATCH -o ./../logs/%j.out
#SBATCH -e ./../logs/%j.err


rm -f ./../logs/latest.out ./../logs/latest.err
ln -s $SLURM_JOBID.out ./../logs/latest.out
ln -s $SLURM_JOBID.err ./../logs/latest.err

echo "START $SLURM_JOBID: $(date)"

module use /appl/local/csc/modulefiles
module load pytorch 


srun python3 join_suomi24.py ../data/splits/suomi24_00.jsonl

echo "END $SLURM_JOBID: $(date)"
