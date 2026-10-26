import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mean magnetization <m>, m = (1/n) sum_i s_i, as a function of temperature: in the minimal-model
# data at T, and in the exact model of each temperature T rescaled to 300 K (exact distribution
# at T' = 300/T, Exact_distribution_T<300/T>.npz).

parser = argparse.ArgumentParser(description='Magnetization in the data and in the rescaled models vs temperature.')
parser.add_argument('--temps', type=int, nargs='+', default=list(range(300, 352, 3)), help='Temperatures (K)')
parser.add_argument('--data_dir', type=str, default='data/MinimalModelData', help='Folder containing DHFR_<T>K')
parser.add_argument('--res_dir', type=str, default='results/MinimalModelModels', help='Folder containing DHFR_<T>K models')
parser.add_argument('--T_target', type=float, default=300, help='Temperature (K) the models are rescaled to')
parser.add_argument('--out', type=str, default='results/MinimalModelAnalysis/Images/magnetization_vs_T.png')
args = parser.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

m_data, err, m_resc = [], [], []
for T in args.temps:
    m = np.load(f"{args.data_dir}/DHFR_{T}K/discretized_array_min_{T}K.npy").mean(1)  # per frame
    m_data.append(m.mean())
    err.append(m.std(ddof=1) / np.sqrt(len(m)))  # assumes independent frames
    d = np.load(f"{args.res_dir}/DHFR_{T}K/Exact_distribution_T{args.T_target / T:.4f}.npz")
    m_resc.append(d['mean'].mean())
    print(f"{T}K  <m> data = {m_data[-1]:.4f} ± {err[-1]:.4f}   rescaled to {args.T_target:g} K = {m_resc[-1]:.4f}")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.errorbar(args.temps, m_data, yerr=err, color='tab:blue', lw=2, marker='o', ms=8, capsize=3,
            label='Data at T')
ax.plot(args.temps, m_resc, color='tab:orange', lw=2, marker='s', ms=8,
        label=f'Exact model of T rescaled to {args.T_target:g} K')
ax.set_xlabel('Temperature T (K)', fontsize=13)
ax.set_ylabel(r'Magnetization $\langle m \rangle$', fontsize=13)
ax.set_title(r'Minimal model: $m = \frac{1}{n}\sum_i s_i$', fontsize=12)
ax.legend(fontsize=10, frameon=False)
ax.grid(True, color='0.9')
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(args.out, dpi=200)
print(f"Saved → {args.out}")
