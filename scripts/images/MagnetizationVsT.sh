#!/usr/bin/env bash
#SBATCH --job-name=MagVsT
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out

# Run from the repo root: sbatch scripts/images/MagnetizationVsT.sh [--out ...]
source env_SBM/bin/activate
env_SBM/bin/python scripts/images/MagnetizationVsT.py "$@"
