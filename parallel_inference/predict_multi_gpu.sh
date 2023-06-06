#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=63
#SBATCH --mem=80G
#SBATCH -p small-g
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:mi250:8
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000119
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOBID.out logs/latest.out
ln -s $SLURM_JOBID.err logs/latest.err

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$@"
echo "START $SLURM_JOBID: $(date)"

srun launch_torch.sh \
    predict_multi_gpu.py "$@"

echo "END $SLURM_JOBID: $(date)"rluukkon@uan04:~/scratch_462000185/risto/torch-registerlabeling> 
