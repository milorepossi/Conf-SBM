#!/usr/bin/env bash
#SBATCH --job-name=ExactDistAll
#SBATCH --array=300-348:3
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%A_%a.out

# Run from the repo root: sbatch scripts/demo-SBM-CM-family/ExactDistributionAll.sh
# For each model DHFR_<T_mod>K: exact distribution at T = 1 and rescaled to 300 K (T = 300/T_mod).
# 351K is already done; at 300K the rescaled distribution is the T = 1 one, so it is computed once.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source env_SBM/bin/activate
T_mod=$SLURM_ARRAY_TASK_ID

env_SBM/bin/python scripts/demo-SBM-CM-family/ExactDistribution.py --T_mod $T_mod --T 1
if [ "$T_mod" -ne 300 ]; then
    env_SBM/bin/python scripts/demo-SBM-CM-family/ExactDistribution.py --T_mod $T_mod --T $(env_SBM/bin/python -c "print(300/$T_mod)")
fi
