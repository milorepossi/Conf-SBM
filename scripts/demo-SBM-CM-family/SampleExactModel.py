import argparse
import numpy as np
from SBM.utils.SimplifiedIsingSampler import Create_modAlign

# Sample the exact Ising model of one temperature (Exact_h.npy / Exact_J.npy, ±1 convention:
# P(s) ∝ exp(sum_i h_i s_i + sum_{i<j} J_ij s_i s_j)) with Create_modAlign, which works with
# {0,1} states. With s = 2a - 1 the same model reads J01 = 4 J, h01_i = 2 h_i - 2 sum_j J_ij.

parser = argparse.ArgumentParser(description='Sample the exact Ising model with Create_modAlign.')
parser.add_argument('--T_mod', type=int, default=351, help='Temperature (K) of the model to sample')
parser.add_argument('--T', type=float, default=300/351, help='Sampling temperature (J, h are divided by it)')
parser.add_argument('--delta_t', type=int, default=8000000, help='MCMC steps (single-site attempts) per chain')
parser.add_argument('--N', type=int, default=35001, help='Number of chains (= number of samples)')
parser.add_argument('--res_dir', type=str, default='results/MinimalModelModels', help='Folder containing DHFR_<T_mod>K')
parser.add_argument('--data_dir', type=str, default='data/MinimalModelData', help='Data used for a quick comparison')
args = parser.parse_args()

folder = f"{args.res_dir}/DHFR_{args.T_mod}K"
h = np.load(f"{folder}/Exact_h.npy")
J = np.load(f"{folder}/Exact_J.npy")
np.fill_diagonal(J, 0)

J01 = 4 * J
h01 = 2 * h - 2 * J.sum(1)

print(f"Sampling DHFR_{args.T_mod}K model at T = {args.T:.4f}, delta_t = {args.delta_t}, N = {args.N}", flush=True)
X01 = Create_modAlign(J01, h01, args.N, delta_t=args.delta_t, temperature=args.T)
S = 2 * X01 - 1  # back to the ±1 convention of the data

out = f"{folder}/Sampled_T{args.T:.4f}_dt{args.delta_t}_N{args.N}.npy"
np.save(out, S)
print(f"Saved → {out}  shape={S.shape}")

# Quick check against the data: single-site means <s_i> and pair correlations <s_i s_j>
mean_S = S.mean(0)
iu = np.triu_indices(S.shape[1], k=1)
C_S = (S.T @ S / S.shape[0])[iu]
for Td in (args.T_mod, 300):
    D = np.load(f"{args.data_dir}/DHFR_{Td}K/discretized_array_min_{Td}K.npy")
    C_D = (D.T @ D / D.shape[0])[iu]
    r = np.corrcoef(mean_S, D.mean(0))[0, 1]
    print(f"<s_i> samples vs {Td}K data: Pearson = {r:.3f}, max |diff| = {np.abs(mean_S - D.mean(0)).max():.3f}")
    r = np.corrcoef(C_S, C_D)[0, 1]
    print(f"<s_i s_j> samples vs {Td}K data: Pearson = {r:.3f}, max |diff| = {np.abs(C_S - C_D).max():.3f}")
