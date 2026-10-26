#!/usr/bin/env bash
#SBATCH --job-name=PlotExact
#SBATCH --array=300-351:3
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%A_%a.out

# Run from the repo root once the ExactInference array has been submitted:
#   sbatch --dependency=aftercorr:<ExactInference job ID> scripts/demo-SBM-CM-family/PlotExactModel.sh
# aftercorr starts each temperature as soon as the matching ExactInference task succeeds.
source env_SBM/bin/activate
env_SBM/bin/python scripts/demo-SBM-CM-family/PlotExactModel.py --T $SLURM_ARRAY_TASK_ID
