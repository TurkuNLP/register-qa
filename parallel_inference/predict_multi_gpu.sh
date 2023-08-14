#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=63 #63 is the whole node on lumi, 40 on puhti # 8 for one gpu
#SBATCH --mem=80G #80G # not sure what would be a good amount on puhti # 16g for one gpu
#SBATCH -p small-g # gpu on puhti
#SBATCH -t 02:00:00 # 1 hour
#SBATCH --gres=gpu:mi250:8 #8 # this means 8 gpu's (the whole node's gpus on lumi), there are 4 on puhti on one node (another way to use gpus on lumi #SBATCH --gpus-per-node=8  )
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000321 #project_2005092 register-labeling on puhti
#SBATCH -o ./../logs/%j.out
#SBATCH -e ./../logs/%j.err

# 8 gpus = 16 gdc's
# you will be billed at a 0.5 rate per GCD allocated. However, if you allocate more than 8 CPU cores or more than 64 GB of memory per GCD you will be billed per slice of 8 cores or 64 GB of memory.
# -> running files is not super expensive and the register project has insane amount of gpu hours

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
