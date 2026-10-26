import argparse
import numpy as np

# Full equilibrium distribution of the exact Ising model of one temperature, by enumerating all
# 2^n configurations. Same convention as ising_mle (ExactInference.py):
#   P(s) = exp(E(s)/T) / Z,   E(s) = sum_i h_i s_i + sum_{i<j} J_ij s_i s_j,   s_i in {-1, +1}.
# Configuration index k <-> s_i = 2*((k >> i) & 1) - 1, i.e. bit i of k is variable i
# (the order of labels_min.npy), bit = 1 meaning s_i = +1.

parser = argparse.ArgumentParser(description='Exact equilibrium distribution of the exact Ising model.')
parser.add_argument('--T_mod', type=int, default=351, help='Temperature (K) of the model')
parser.add_argument('--T', type=float, default=1.0, help='Temperature at which the distribution is computed (J, h are divided by it)')
parser.add_argument('--res_dir', type=str, default='results/MinimalModelModels', help='Folder containing DHFR_<T_mod>K')
parser.add_argument('--data_dir', type=str, default='data/MinimalModelData', help='Data used for a quick check at T = 1')
args = parser.parse_args()

folder = f"{args.res_dir}/DHFR_{args.T_mod}K"
h = np.load(f"{folder}/Exact_h.npy")
J = np.load(f"{folder}/Exact_J.npy")
np.fill_diagonal(J, 0)
n = len(h)

k = np.arange(1 << n, dtype=np.int64)
cfg = ((k[:, None] >> np.arange(n)) & 1).astype(np.float64) * 2 - 1  # (2^n, n)

E = cfg @ h + ((cfg @ J) * cfg).sum(1) / 2
logw = E / args.T
logZ = logw.max() + np.log(np.exp(logw - logw.max()).sum())
P = np.exp(logw - logZ)

mean = P @ cfg
C = (cfg.T * P) @ cfg  # <s_i s_j>, ones on the diagonal

out = f"{folder}/Exact_distribution_T{args.T:.4f}.npz"
np.savez(out, P=P, E=E, logZ=logZ, mean=mean, C=C, T=args.T, T_mod=args.T_mod,
         encoding="index k <-> s_i = 2*((k >> i) & 1) - 1; P(s) = exp(E(s)/T)/Z, E = h.s + sum_{i<j} J_ij s_i s_j")
print(f"Saved → {out}")
print(f"n = {n}, states = {len(P)}, sum P = {P.sum():.12f}, log Z = {logZ:.6f}")
print(f"entropy = {-(P * np.log(P)).sum():.4f} nats, most probable state P = {P.max():.4f}")

if abs(args.T - 1) < 1e-12:
    # The exact MLE matches the data moments, so at T = 1 these differences must be ~0
    D = np.load(f"{args.data_dir}/DHFR_{args.T_mod}K/discretized_array_min_{args.T_mod}K.npy")
    iu = np.triu_indices(n, k=1)
    print(f"max |<s_i> exact - data|     = {np.abs(mean - D.mean(0)).max():.2e}")
    print(f"max |<s_i s_j> exact - data| = {np.abs(C[iu] - (D.T @ D / len(D))[iu]).max():.2e}")
