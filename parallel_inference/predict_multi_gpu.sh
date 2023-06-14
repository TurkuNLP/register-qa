#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10 #63 # this is the whole node?
#SBATCH --mem=64G
#SBATCH -p small-g # gpu on puhti
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:mi250:1 #8 # this means 8 gpu's (the whole node)
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000241 #project_2005092 register-labeling on puhti
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
# module load CrayEnv
# module load cray-python

# $@ gives all the arguments from the other sh script

srun python3 predict_multi_gpu.py "$@" ##launch_torch.sh \

echo "END $SLURM_JOBID: $(date)" #rluukkon@uan04:~/scratch_462000185/risto/torch-registerlabeling> 
