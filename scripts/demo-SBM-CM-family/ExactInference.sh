#!/usr/bin/env bash
#SBATCH --job-name=ExactInf
#SBATCH --array=300-351:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%A_%a.out

# Run from the repo root: sbatch scripts/demo-SBM-CM-family/ExactInference.sh
# One array task per temperature (300K to 351K in steps of 3K).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source env_SBM/bin/activate
env_SBM/bin/python scripts/demo-SBM-CM-family/ExactInference.py --T $SLURM_ARRAY_TASK_ID
