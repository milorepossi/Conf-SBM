import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Single-site and pairwise state frequencies of model samples vs data (both ±1 convention),
# in the style of results/AlloCommModels/DHFR_351K/rescaled_vs_300K_351K_stacking_dt8000000.png.
# State 0 is s = -1 and state 1 is s = +1 (the original 0/1 discretization).

parser = argparse.ArgumentParser(description='Compare state frequencies of samples and data.')
parser.add_argument('--T', type=float, default=300/351, help='Sampling temperature of the 351K model; sets the defaults below')
parser.add_argument('--samples', type=str, default=None)
parser.add_argument('--data', type=str, default='data/MinimalModelData/DHFR_300K/discretized_array_min_300K.npy')
parser.add_argument('--labels', type=str, default='data/MinimalModelData/labels_min.npy')
parser.add_argument('--highlight', type=str, default='WT_stacking_ligand_cofactor_300K', help='Variable marked with a star')
parser.add_argument('--title', type=str, default=None)
parser.add_argument('--xlabel', type=str, default='Frequencies at 300K')
parser.add_argument('--ylabel', type=str, default=None)
parser.add_argument('--out', type=str, default=None)
args = parser.parse_args()

res = 'results/MinimalModelModels/DHFR_351K'
rescaled = abs(args.T - 300/351) < 1e-4
if args.samples is None:
    args.samples = f'{res}/Sampled_T{args.T:.4f}_dt8000000_N35001.npy'
if args.title is None:
    args.title = ('Rescaled 351K exact model VS 300K data (delta_t = 8e+06)' if rescaled else
                  f'351K exact model sampled at T = {args.T:g} VS 300K data (delta_t = 8e+06)')
if args.ylabel is None:
    args.ylabel = ('Frequencies through rescaling\n351K exact model sampled at T = 300/351' if rescaled else
                   f'Frequencies of the 351K exact model\nsampled at T = {args.T:g}')
if args.out is None:
    args.out = (f'{res}/rescaled_vs_300K_351K_stacking_dt8000000.png' if rescaled else
                f'{res}/T{args.T:.4f}_vs_300K_351K_stacking_dt8000000.png')


def frequencies(X):
    """One-hot (state 0 = -1, state 1 = +1) single-site (L, 2) and pairwise (L, L, 2, 2) frequencies."""
    oh = np.stack([(X == -1), (X == 1)], axis=2).astype(np.float64)  # (N, L, 2)
    f1 = oh.mean(0)
    f2 = np.einsum('nia,njb->ijab', oh, oh) / X.shape[0]
    return f1, f2


S = np.load(args.samples)
D = np.load(args.data)
labels = [str(l) for l in np.load(args.labels)]
L = S.shape[1]
assert D.shape[1] == L and len(labels) == L
assert set(np.unique(S)) <= {-1, 1} and set(np.unique(D)) <= {-1, 1}

f1_S, f2_S = frequencies(S)
f1_D, f2_D = frequencies(D)
iu = np.triu_indices(L, k=1)
p_S, p_D = f2_S[iu].ravel(), f2_D[iu].ravel()
s_S, s_D = f1_S.ravel(), f1_D.ravel()
r_pair = np.corrcoef(p_D, p_S)[0, 1]
r_single = np.corrcoef(s_D, s_S)[0, 1]

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(p_D, p_S, s=12, color='tab:orange', alpha=0.4,
           label=f'Pairwise frequencies\n Pearson: {r_pair:.3f}')
ax.scatter(s_D, s_S, s=12, color='tab:blue',
           label=f'Single-site frequencies\n Pearson: {r_single:.3f}')

k = labels.index(args.highlight)
ax.scatter(f1_D[k], f1_S[k], s=400, marker='*', color='tab:red', edgecolor='black', zorder=5,
           label=args.highlight.removesuffix('_300K'))
for a in range(2):
    ax.annotate(f'state {a}', (f1_D[k, a], f1_S[k, a]), xytext=(12, -14), textcoords='offset points',
                color='tab:red', fontsize=12)

ax.plot([0, 1], [0, 1], color='black', lw=1.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlabel(args.xlabel, fontsize=16)
ax.set_ylabel(args.ylabel, fontsize=14)
ax.set_title(args.title, fontsize=13)
ax.legend(loc='upper left', fontsize=12)
fig.tight_layout()
fig.savefig(args.out, dpi=100)
print(f"Pearson pairwise = {r_pair:.3f}, single-site = {r_single:.3f}")
print(f"{args.highlight}: data {f1_D[k]}, samples {f1_S[k]}")
print(f"Saved → {args.out}")
