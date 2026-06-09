import numpy as np
from pathlib import Path
import SBM
import SBM.utils.utils as ut
import SBM.utils.utils_plot as up

ROOT = Path(SBM.__file__).resolve().parents[2] 
data_dir = ROOT / "data"
results_dir = ROOT / "results"

file = results_dir/'DHFR/DHFR_ModelBM_N_chains1000_N_iter1000_Param_initZero_k_MCMC100000_lambda_J0.001_lambda_h0.001_m1_theta0_N_Av1_R0.npy'
output = np.load(file,allow_pickle=True)[()]
align_mod = ut.Create_modAlign(output,output['align'].shape[0],delta_t = output['options0']['k_MCMC'],temperature=1)
#align_mod = ut.Create_modAlign(output,100,delta_t = output['options0']['k_MCMC'],temperature=0.1)

output['align_mod'] = align_mod

Stats = ut.compute_stats(output,align_mod)

up.plot_stats(output,Stats,plot = 'Freq')
up.plot_stats(output,Stats,plot = 'Pair_freq',ma = 0.25)
up.plot_stats(output,Stats,plot = 'PCA')