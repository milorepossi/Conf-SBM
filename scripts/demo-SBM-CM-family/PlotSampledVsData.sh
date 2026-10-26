#!/usr/bin/env bash
#SBATCH --job-name=PlotSampled
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out

# Run from the repo root: sbatch scripts/demo-SBM-CM-family/PlotSampledVsData.sh [--T 1 ...]
# Defaults compare the 351K exact model sampled at T = 300/351 with the 300K data.
source env_SBM/bin/activate
env_SBM/bin/python scripts/demo-SBM-CM-family/PlotSampledVsData.py "$@"
