#!/usr/bin/env bash
#SBATCH --job-name=ExactDist
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

# Run from the repo root: sbatch scripts/demo-SBM-CM-family/ExactDistribution.sh [--T 0.8547008547008547 ...]
# Extra arguments override the defaults (351K model at T = 1).
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source env_SBM/bin/activate
env_SBM/bin/python scripts/demo-SBM-CM-family/ExactDistribution.py --T_mod 351 --T 1 "$@"
