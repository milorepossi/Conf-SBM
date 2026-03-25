#!/usr/bin/env bash
source env_SBM/bin/activate
env_SBM/bin/python scripts/demo-SBM-CM-family/SBM-CM-family.py --Input_MSA data/Conformations/DHFR.npy --Weights data/Conformations_Weights/w01_DHFR.npy --fam DHFR --mod BM --N_chains 1000 --TestTrain 0 --k_MCMC 100000 --lambdJ 0.001 --lambdh 0.001 --N_av 1 --N_iter 1000 --rep 1 --ParamInit profile