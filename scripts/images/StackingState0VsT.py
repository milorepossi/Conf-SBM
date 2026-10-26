import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Probability of state 0 (s = -1) of the ligand-cofactor stacking variable as a function of
# temperature: in the minimal-model data, and in the exact model of each temperature T rescaled
# to 300 K (exact distribution at T' = 300/T, Exact_distribution_T<300/T>.npz).

parser = argparse.ArgumentParser(description='P(stacking state 0) in the data vs temperature.')
parser.add_argument('--temps', type=int, nargs='+', default=list(range(300, 352, 3)), help='Temperatures (K)')
parser.add_argument('--data_dir', type=str, default='data/MinimalModelData', help='Folder containing DHFR_<T>K')
parser.add_argument('--res_dir', type=str, default='results/MinimalModelModels', help='Folder containing DHFR_<T>K models')
parser.add_argument('--T_target', type=float, default=300, help='Temperature (K) the models are rescaled to')
parser.add_argument('--labels', type=str, default='data/MinimalModelData/labels_min.npy')
parser.add_argument('--var', type=str, default='WT_stacking_ligand_cofactor_300K', help='Variable to plot')
parser.add_argument('--out', type=str, default='results/MinimalModelAnalysis/Images/stacking_state0_vs_T_data.png')
args = parser.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

labels = [str(l) for l in np.load(args.labels)]
k = labels.index(args.var)

p0, err = [], []
for T in args.temps:
    s = np.load(f"{args.data_dir}/DHFR_{T}K/discretized_array_min_{T}K.npy")[:, k]
    p = np.mean(s == -1)
    p0.append(p)
    err.append(np.sqrt(p * (1 - p) / len(s)))  # binomial, assumes independent frames
    print(f"{T}K  P(state 0) = {p:.4f} ± {err[-1]:.4f}  (N = {len(s)})")
p0, err = np.array(p0), np.array(err)

p0_resc = []
for T in args.temps:
    d = np.load(f"{args.res_dir}/DHFR_{T}K/Exact_distribution_T{args.T_target / T:.4f}.npz")
    p0_resc.append((1 - d['mean'][k]) / 2)  # P(s = -1) from <s>
    print(f"{T}K model rescaled to {args.T_target:g} K  P(state 0) = {p0_resc[-1]:.4f}")
p0_resc = np.array(p0_resc)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(args.temps, p0, yerr=err, color='tab:blue', lw=2, marker='o', ms=8, capsize=3,
            label='Data at T')
ax.plot(args.temps, p0_resc, color='tab:orange', lw=2, marker='s', ms=8,
        label=f'Exact model of T rescaled to {args.T_target:g} K')
ax.set_xlabel('Temperature T (K)', fontsize=13)
ax.set_ylabel('P(state 0)', fontsize=13)
ax.set_title(f"{args.var.removesuffix('_300K')}: state 0", fontsize=12)
ax.legend(fontsize=10, frameon=False)
ax.grid(True, color='0.9')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(args.out, dpi=200)
print(f"Saved → {args.out}")
