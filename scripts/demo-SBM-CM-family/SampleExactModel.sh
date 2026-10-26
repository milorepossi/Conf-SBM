#!/usr/bin/env bash
#SBATCH --job-name=SampleExact
#SBATCH --cpus-per-task=50
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

# Run from the repo root: sbatch scripts/demo-SBM-CM-family/SampleExactModel.sh [--T 1 ...]
# Extra arguments override the defaults below.
# 50 CPUs because the MCMC kernel hard-codes omp_set_num_threads(50).
source env_SBM/bin/activate
env_SBM/bin/python scripts/demo-SBM-CM-family/SampleExactModel.py --T_mod 351 --T 0.8547008547008547 --delta_t 8000000 --N 35001 "$@"
